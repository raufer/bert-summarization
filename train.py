import os
import json
import math
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import backend as K
from tensorflow.python.keras.initializers import Constant

from tqdm import tqdm
from config import config
from arguments import args

from data.load import load_cnn_dailymail

from random import randint
from rouge import Rouge

from ops.tokenization import tokenizer
from ops.tokenization import convert_idx_to_token_tensor

from ops.session import initialize_vars
from ops.session import save_variable_specs

from ops.metrics import calculate_rouge
from ops.tensor import with_column
from ops.regularization import label_smoothing
from ops.optimization import noam_scheme

from models.abstractive_summarizer import AbstractiveSummarization
from models.abstractive_summarizer import train
from models.abstractive_summarizer import eval


logger = logging.getLogger()
logger.setLevel(logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO) 
tf.enable_resource_variables()

logging.info('Job Configuration:\n' + str(config))   

    
model = AbstractiveSummarization(
    num_layers=config.NUM_LAYERS,
    d_model=config.D_MODEL,
    num_heads=config.NUM_HEADS,
    dff=config.D_FF,
    vocab_size=config.VOCAB_SIZE,
    input_seq_len=config.INPUT_SEQ_LEN,
    output_seq_len=config.OUTPUT_SEQ_LEN,
    rate=config.DROPOUT_RATE
)


train_dataset, val_dataset, test_dataset, n_train_examples, n_val_examples, n_test_examples = load_cnn_dailymail()

n_train_batches = n_train_examples // config.BATCH_SIZE
n_val_batches = n_val_examples // config.BATCH_SIZE
n_test_batches = n_test_examples // config.BATCH_SIZE

logging.info(f"'{n_train_examples}' training examples, '{n_train_batches}' batches")
logging.info(f"'{n_val_examples}' validation examples, '{n_val_batches}' batches")
logging.info(f"'{n_test_examples}' testing examples, '{n_test_batches}' batches")


train_iterator = train_dataset.make_initializable_iterator()
train_stream = train_iterator.get_next()

xs, ys = train_stream[:3], train_stream[3:]
train_loss, zero_op, accumlation_op, train_op, global_step, train_summaries = train(model, xs, ys, gradient_accumulation=True)

if args.eval:
    val_iterator = val_dataset.make_initializable_iterator()
    val_stream = val_iterator.get_next()

    xs, ys = val_stream[:3], val_stream[3:]    
    y, y_hat, eval_loss, eval_summaries = eval(model, xs, ys)

saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS)

# config_tf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config_tf = tf.ConfigProto(allow_soft_placement=True)
config_tf.gpu_options.allow_growth=True

run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)


with tf.Session(config=config_tf) as sess:

    if os.path.isdir(config.LOGDIR):
        shutil.rmtree(config.LOGDIR)

    os.mkdir(config.LOGDIR)

    ckpt = tf.train.latest_checkpoint(config.CHECKPOINTDIR)

    rouge = Rouge()

    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(config.LOGDIR, "specs"))
    else:
        saver.restore(sess, ckpt)        

    summary_writer_train = tf.summary.FileWriter(os.path.join(config.LOGDIR), sess.graph)
    summary_writer_eval = tf.summary.FileWriter(os.path.join(config.LOGDIR, 'eval'), sess.graph)

    initialize_vars(sess)

    _gs = sess.run(global_step)

    sess.run(train_iterator.initializer)
    if args.eval:
        sess.run(val_iterator.initializer)

    total_steps = int(config.NUM_EPOCHS * (n_train_batches / config.GRADIENT_ACCUMULATION_N_STEPS))

    logger.info(f"Running Training Job for '{total_steps}' steps")
    
    for i in tqdm(range(_gs, total_steps+1)):
        
        #  gradient accumulation mechanism
        sess.run(zero_op)
        
        for i in range(config.GRADIENT_ACCUMULATION_N_STEPS):
            sess.run(accumlation_op)

        _loss, _, _gs, _summary = sess.run([train_loss, train_op, global_step, train_summaries], options=run_options)

        epoch = math.ceil(_gs / n_train_batches)

        summary_writer_train.add_summary(_summary, _gs)
        summary_writer_train.flush() 

        if (_gs % n_train_batches == 0):
            
            if args.eval:

                logger.info(f"Epoch '{epoch}' done")
                logger.info(f"Current training step: '{_gs}")

                _y, _y_hat, _eval_summary = sess.run([y, y_hat, eval_summaries])

                summary_writer_eval.add_summary(_eval_summary, 0)
                summary_writer_eval.flush()       

                # monitor a random sample
                rnd = randint(0, _y.shape[0] - 1)

                y_rnd = ' '.join(tokenizer.convert_ids_to_tokens(_y[rnd]))
                y_hat_rnd = ' '.join(tokenizer.convert_ids_to_tokens(_y_hat[rnd]))

                rouges = rouge.get_scores(y_rnd, y_hat_rnd)[0]
                r1_val, r2_val, rl_val = rouges['rouge-1']["f"], rouges['rouge-2']["f"], rouges['rouge-l']["f"]

                print('Target:')
                print(y_rnd)
                print('Prediction:')
                print(y_hat_rnd)

                print(f"ROUGE-1 '{r1_val}'")
                print(f"ROUGE-2 '{r2_val}'")
                print(f"ROUGE-L '{rl_val}'")
                print(f"ROUGE-AVG '{np.mean([r1_val, r2_val, rl_val])}'", '\n--\n')

            logging.info("Checkpoint: Saving Model")

            model_output = f"abstractive_summarization_2019_epoch_{epoch}_loss_{str(round(_loss, 4))}"

            ckpt_name = os.path.join(config.CHECKPOINTDIR, model_output)

            saver.save(sess, ckpt_name, global_step=_gs)

            logging.info(f"After training '{_gs}' steps, '{ckpt_name}' has been saved.")
            
    model_output = f"abstractive_summarization_2019_final"
    ckpt_name = os.path.join(config.CHECKPOINTDIR, model_output)
    saver.save(sess, ckpt_name, global_step=_gs)            

    summary_writer_train.close()  
    summary_writer_eval.close()
    