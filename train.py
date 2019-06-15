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

from data.load import load_cnn_dailymail

from data.load import UNK_ID
from data.load import CLS_ID
from data.load import SEP_ID
from data.load import MASK_ID

from random import randint
from rouge import Rouge

from layers.transformer import Encoder
from layers.transformer import Decoder
from layers.bert import BertLayer, BERT_MODEL_URL
from layers.transformer import Decoder

from ops.masking import create_masks
from ops.masking import create_look_ahead_mask
from ops.masking import create_padding_mask
from ops.masking import mask_timestamp
from ops.masking import tile_and_mask_diagonal

from ops.tokenization import tokenizer
from ops.tokenization import convert_idx_to_token_tensor

from ops.session import initialize_vars
from ops.session import save_variable_specs

from ops.metrics import calculate_rouge
from ops.tensor import with_column
from ops.regularization import label_smoothing
from ops.optimization import noam_scheme


logger = logging.getLogger()
logger.setLevel(logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO) 
tf.enable_resource_variables()


warmup_steps = config.WARMUP_STEPS
initial_lr = config.INITIAL_LR

BERT_MODEL_URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

logging.info('Job Configuration:\n' + str(config))

train_dataset, val_dataset, test_dataset, n_train_examples, n_val_examples, n_test_examples = load_cnn_dailymail()

n_train_batches = n_train_examples // config.BATCH_SIZE
n_val_batches = n_val_examples // config.BATCH_SIZE
n_test_batches = n_test_examples // config.BATCH_SIZE

logging.info(f"'{n_train_examples}' training examples, '{n_train_batches}' batches")
logging.info(f"'{n_val_examples}' validation examples, '{n_val_batches}' batches")
logging.info(f"'{n_test_examples}' testing examples, '{n_test_batches}' batches")


def _embedding_from_bert():
    """
    Extract the preratined word embeddings from a BERT model
    Returns a numpy matrix with the embeddings
    """
    logger.info("Extracting pretrained word embeddings weights from BERT")
    
    bert = hub.Module(BERT_MODEL_URL, trainable=False, name="embeddings_from_bert_module")    
    
    with tf.Session() as sess:
        initialize_vars(sess)
        embedding_matrix = sess.run(bert.variable_map['bert/embeddings/word_embeddings'])
                
    logger.info(f"Embedding matrix shape '{embedding_matrix.shape}'")
    return embedding_matrix


class AbstractiveSummarization(tf.keras.Model):
    """
    Pretraining-Based Natural Language Generation for Text Summarization 
    https://arxiv.org/pdf/1902.09243.pdf
    """
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, input_seq_len, output_seq_len, rate=0.1):
        super(AbstractiveSummarization, self).__init__()
        
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        self.vocab_size = vocab_size
    
        self.bert = BertLayer(seq_len=seq_len, d_embedding=d_model, trainable=False)
        
        embedding_matrix = _embedding_from_bert()
        
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, trainable=False,
            embeddings_initializer=Constant(embedding_matrix)
        )
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
                
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def encode(self, ids, mask, segment_ids):
        # (batch_size, seq_len, d_bert)
        return self.bert((ids, mask, segment_ids))
    
    def draft_summary(self, enc_output, look_ahead_mask, padding_mask, target_ids=None, training=True):
        
        logging.info("Building:'Draft summary'")
            
        # (batch_size, seq_len)
        dec_input = target_ids
                
        # (batch_size, seq_len, d_bert)
        embeddings = self.embedding(target_ids) 

        # (batch_size, seq_len, d_bert), (_)            
        dec_output, attention_dist = self.decoder(embeddings, enc_output, training, look_ahead_mask, padding_mask)
        
        # (batch_size, seq_len, vocab_len)
        logits = self.final_layer(dec_output)
        
        # (batch_size, seq_len)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))
        
        return logits, preds, attention_dist      
    
    def draft_summary_greedy(self, enc_output, look_ahead_mask, padding_mask, training=False):
        """
        Inference call, builds a draft summary autoregressively
        """
        
        logging.info("Building: 'Greedy Draft Summary'")
        
        N = tf.shape(enc_output)[0]
        T = tf.shape(enc_output)[1]

        # (batch_size, 1)
        dec_input = tf.ones([N, 1], dtype=tf.int32) * CLS_ID
            
        summary, dec_outputs, dec_logits, attention_dists = [], [], [], []
        
        summary += [dec_input]
        dec_logits += [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.vocab_size), axis=0), [N, 1, 1])]
    
        for i in tqdm(range(0, self.output_seq_len - 1)):
                    
            # (batch_size, i+1, d_bert)
            embeddings = self.embedding(dec_input)    
            
            # (batch_size, i+1, d_bert), (_)            
            dec_output, attention_dist = self.decoder(embeddings, enc_output, training, look_ahead_mask, padding_mask)
            
            # (batch_size, 1, d_bert)
            dec_output_i = dec_output[:, -1: ,:]

            # (batch_size, 1, d_bert)
            dec_outputs += [dec_output_i]
            attention_dists += [{k: v[:, -1:, :] for k, v in attention_dist.items()}]
        
            # (batch_size, 1, vocab_len)
            logits = self.final_layer(dec_output_i)
            
            dec_logits += [logits]
        
            # (batch_size, 1)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            
            summary += [preds]
            
            # (batch_size, i+2)
            dec_input = with_column(dec_input, i+1, preds)
            
        # (batch_size, seq_len, vocab_len)            
        dec_logits = tf.concat(dec_logits, axis=1)
        
        # (batch_size, seq_len)            
        summary = tf.concat(summary, axis=1)                    
        
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
        return dec_logits, summary, attention_dists    
    
    def refined_summary_iter(self, enc_output, target, padding_mask, training=True):
        """
        Iterative version of refined summary using teacher forcing
        """
        
        logging.info("Building: 'Refined Summary'")  
        
        N = tf.shape(enc_output)[0]            
        
        dec_inp_ids, dec_inp_mask, dec_inp_segment_ids = target
        
        dec_outputs, attention_dists = [], []

        for i in tqdm(range(1, self.output_seq_len)):
            
            # (batch_size, seq_len)
            dec_inp_ids_ = mask_timestamp(dec_inp_ids, i, MASK_ID)
            
            # (batch_size, seq_len, d_bert)
            context_vectors = self.bert((dec_inp_ids_, dec_inp_mask, dec_inp_segment_ids))
            
            # (batch_size, seq_len, d_bert), (_)
            dec_output, attention_dist = self.decoder(
                context_vectors,
                enc_output,
                training,
                look_ahead_mask=None,
                padding_mask=padding_mask
            )
            
            # (batch_size, 1, seq_len)
            dec_outputs += [dec_output[:,i:i+1,:]]
            attention_dists += [{k: v[:, i:i+1, :] for k, v in attention_dist.items()}]

        # (batch_size, seq_len - 1, d_bert)            
        dec_outputs = tf.concat(dec_outputs, axis=1)
                
        # (batch_size, seq_len - 1, vocab_len)
        logits = self.final_layer(dec_outputs)
        
        # (batch_size, seq_len, vocab_len), accommodate for initial [CLS]
        logits = tf.concat(
            [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.vocab_size), axis=0), [N, 1, 1]), logits],
            axis=1
        )
        
        # (batch_size, seq_len)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))
        
        return logits, preds, attention_dists
    
    def refined_summary(self, enc_output, target, padding_mask, training=True):

        logging.info("Building: 'Refined Summary'")              
        
        N = tf.shape(enc_output)[0]
        T = tf.shape(enc_output)[1]

        # (batch_size, seq_len) x3
        dec_inp_ids, dec_inp_mask, dec_inp_segment_ids = target
        
        # since we are using teacher forcing we do not need an autoregressice mechanism here

        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_ids = tile_and_mask_diagonal(dec_inp_ids, mask_with=MASK_ID)
        
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_mask = tf.tile(dec_inp_mask, [T-1, 1])
        
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_segment_ids = tf.tile(dec_inp_segment_ids, [T-1, 1])
        
        # (batch_size x (seq_len - 1), seq_len, d_bert) 
        enc_output = tf.tile(enc_output, [T-1, 1, 1])
        
        # (batch_size x (seq_len - 1), 1, 1, seq_len) 
        padding_mask = tf.tile(padding_mask, [T-1, 1, 1, 1])
                
        # (batch_size x (seq_len - 1), seq_len, d_bert)
        context_vectors = self.bert((dec_inp_ids, dec_inp_mask, dec_inp_segment_ids))   

        # (batch_size x (seq_len - 1), seq_len, d_bert), (_)
        dec_outputs, attention_dists = self.decoder(
            context_vectors,
            enc_output,
            training,
            look_ahead_mask=None,
            padding_mask=padding_mask
        )
                
        # (batch_size x (seq_len - 1), seq_len - 1, d_bert)
        dec_outputs = dec_outputs[:, 1:, :]
        
        # (batch_size x (seq_len - 1), (seq_len - 1))
        diag = tf.linalg.set_diag(tf.zeros([T-1, T-1]), tf.ones([T-1]))
        diag = tf.tile(diag, [N, 1])
        
        where = tf.not_equal(diag, 0)
        indices = tf.where(where)
        
        # (batch_size x (seq_len - 1), d_bert)
        dec_outputs = tf.gather_nd(dec_outputs, indices)
        
        # (batch_size x (seq_len - 1), d_bert)
        dec_outputs = tf.reshape(dec_outputs, [N, T-1, -1])
        
        # (batch_size, seq_len - 1, vocab_len)
        logits = self.final_layer(dec_outputs)
        
        # (batch_size, seq_len, vocab_len), accommodate for initial [CLS]
        logits = tf.concat(
            [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.vocab_size), axis=0), [N, 1, 1]), logits],
            axis=1
        )
        
        # (batch_size, seq_len)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))
        
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)   
        return logits, preds, attention_dists             
    
    def refined_summary_greedy(self, enc_output, draft_summary, padding_mask, training=False):
        """
        Inference call, builds a refined summary
        
        It first masks each word in the summary draft one by one,
        then feeds the draft to BERT to generate context vectors.
        """
        
        logging.info("Building: 'Greedy Refined Summary'")
                
        refined_summary = draft_summary
        refined_summary_mask = tf.cast(tf.math.equal(draft_summary, 0), tf.float32)
        refined_summary_segment_ids = tf.zeros(tf.shape(draft_summary))
                
        N = tf.shape(draft_summary)[0]            
        T = tf.shape(draft_summary)[1]
        
        dec_outputs, dec_logits, attention_dists = [], [], []
        
        dec_logits += [tf.tile(tf.expand_dims(tf.one_hot([CLS_ID], self.vocab_size), axis=0), [N, 1, 1])]

        for i in tqdm(range(1, self.output_seq_len)):
            
            # (batch_size, seq_len)
            refined_summary_ = mask_timestamp(refined_summary, i, MASK_ID)
            
            # (batch_size, seq_len, d_bert)
            context_vectors = self.bert((refined_summary_, refined_summary_mask, refined_summary_segment_ids))
            
            # (batch_size, seq_len, d_bert), (_)
            dec_output, attention_dist = self.decoder(
                context_vectors,
                enc_output,
                training=training,
                look_ahead_mask=None,
                padding_mask=padding_mask
            )
            
            # (batch_size, 1, vocab_len)
            dec_output_i = dec_output[:, i:i+1 ,:]
            
            dec_outputs += [dec_output_i]
            attention_dists += [{k: v[:, i:i+1, :] for k, v in attention_dist.items()}]
            
            # (batch_size, 1, vocab_len)
            logits = self.final_layer(dec_output_i)
            
            dec_logits += [logits]            
        
            # (batch_size, 1)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))            
            
            # (batch_size, seq_len)
            refined_summary = with_column(refined_summary, i, preds)
            
        # (batch_size, seq_len, vocab_len)            
        dec_logits = tf.concat(dec_logits, axis=1)            
        
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)        
        return dec_logits, refined_summary, attention_dists
    
    def call(self, inp, tar=None, training=False):
        """
        Run the model for training/inference
        For training, the target is needed (ids, mask, segments)
        """
        
        if training:
            return self.fit(inp, tar)
        
        else:
            return self.predict(inp)
        
    
    def fit(self, inp, tar):
        """
        __call__ for training; uses teacher forcing for both the draft
        and the defined decoder
        """
        
        # (batch_size, seq_len) x3
        input_ids, input_mask, input_segment_ids = inp
        
        # (batch_size, seq_len + 1) x3
        target_ids, target_mask, target_segment_ids = tar   
                
        # (batch_size, 1, 1, seq_len), (_), (batch_size, 1, 1, seq_len)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids[:, :-1])

        # (batch_size, seq_len, d_bert)
        enc_output = self.encode(input_ids, input_mask, input_segment_ids)

        # (batch_size, seq_len , vocab_len), (batch_size, seq_len), (_)
        logits_draft_summary, preds_draft_summary, draft_attention_dist = self.draft_summary(
            enc_output=enc_output,
            look_ahead_mask=combined_mask,
            padding_mask=dec_padding_mask,
            target_ids=target_ids[:, :-1],
            training=True
        )             

        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
        logits_refined_summary, preds_refined_summary, refined_attention_dist = self.refined_summary(
            enc_output=enc_output,
            target=(target_ids[:, :-1], target_mask[:, :-1], target_segment_ids[:, :-1]),            
            padding_mask=dec_padding_mask,
            training=True
        )

        return logits_draft_summary, logits_refined_summary
    
    
    def predict(self, inp):
        """
        __call__ for inference; uses teacher forcing for both the draft
        and the defined decoder
        """
        
            
        # (batch_size, seq_len) x3
        input_ids, input_mask, input_segment_ids = inp

        dec_padding_mask = create_padding_mask(input_ids)        

        # (batch_size, seq_len, d_bert)
        enc_output = self.encode(input_ids, input_mask, input_segment_ids)
        
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
        logits_draft_summary, preds_draft_summary, draft_attention_dist = self.draft_summary_greedy(
            enc_output=enc_output,
            look_ahead_mask=None,
            padding_mask=dec_padding_mask
        )
                        
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
        logits_refined_summary, preds_refined_summary, refined_attention_dist = self.refined_summary_greedy(
            enc_output=enc_output,
            padding_mask=dec_padding_mask,
            draft_summary=preds_draft_summary
        )
        
        return logits_draft_summary, preds_draft_summary, draft_attention_dist, logits_refined_summary, preds_refined_summary, refined_attention_dist
    
    
    def train(self, xs, ys):
        
        logging.info("Building Training Graph")

        # (batch_size, seq_len + 1) x3
        target_ids, _, _ = ys
            
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len, vocab_len)
        logits_draft_summary, logits_refined_summary = self(xs, ys, True)
        
        target_ids_ = label_smoothing(tf.one_hot(target_ids, depth=self.vocab_size))

        # use right shifted target, (batch_size, seq_len)
        loss_draft = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_draft_summary, labels=target_ids_[:, 1:, :])
        mask = tf.math.logical_not(tf.math.equal(target_ids[:, 1:], 0))    
        mask = tf.cast(mask, dtype=loss_draft.dtype)
        loss_draft *= mask
        
        # use non-shifted target (we want to predict the masked word), (batch_size, seq_len)
        loss_refined = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_refined_summary, labels=target_ids_[:, :-1, :])
        mask = tf.math.logical_not(tf.math.equal(target_ids[:, :-1], 0))    
        mask = tf.cast(mask, dtype=loss_refined.dtype)
        loss_refined *= mask        

        # (batch_size, seq_len)
        loss = loss_draft + loss_refined            
        # scalar
        loss = tf.reduce_mean(loss)

        global_step = tf.train.get_or_create_global_step()
        learning_rate = noam_scheme(initial_lr, global_step, warmup_steps)
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
        train_op = optimizer.minimize(loss, global_step=global_step, colocate_gradients_with_ops=True)

        tf.summary.scalar('learning_rate', learning_rate, family='train')
        tf.summary.scalar('loss_draft', tf.reduce_mean(loss_draft * mask), family='train')
        tf.summary.scalar('loss_refined', tf.reduce_mean(loss_refined * mask), family='train')
        tf.summary.scalar("loss", loss, family='train')
        tf.summary.scalar("global_step", global_step, family='train')

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries
    
    def eval(self, xs, ys):
        
        logging.info("Building Evaluation Graph")
            
        # (batch_size, seq_len + 1) x3
        target_ids, _, _ = ys
                        
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_), (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
        logits_draft_summary, preds_draft_summary, _, logits_refined_summary, preds_refined_summary, _ = self(xs)
        
        target_ids_ = label_smoothing(tf.one_hot(target_ids, depth=self.vocab_size))
        
        # use right shifted target, (batch_size, seq_len)
        loss_draft = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_draft_summary, labels=target_ids_[:, 1:, :])
        mask = tf.math.logical_not(tf.math.equal(target_ids[:, 1:], 0))    
        mask = tf.cast(mask, dtype=loss_draft.dtype)
        loss_draft *= mask
        
        # use non-shifted target (we want to predict the masked word), (batch_size, seq_len)
        loss_refined = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_refined_summary, labels=target_ids_[:, :-1, :])
        mask = tf.math.logical_not(tf.math.equal(target_ids[:, :-1], 0))    
        mask = tf.cast(mask, dtype=loss_refined.dtype)
        loss_refined *= mask        

        # (batch_size, seq_len)
        loss = loss_draft + loss_refined
        
        # scalar
        loss = tf.reduce_mean(loss)
        
        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(xs[0])[0] - 1, tf.int32)
        
        x_rnd = convert_idx_to_token_tensor(xs[0][n])
        y_rnd = convert_idx_to_token_tensor(target_ids[n, :-1])
        y_hat_rnd = convert_idx_to_token_tensor(preds_refined_summary[n])
        
        r1_val, r2_val, rl_val, r_vag = calculate_rouge(y_rnd, y_hat_rnd)

        tf.summary.text("input", x_rnd)
        tf.summary.text("target", y_rnd)
        tf.summary.text("prediction", y_hat_rnd)
        
        tf.summary.scalar('ROUGE-1', r1_val, family='eval')
        tf.summary.scalar('ROUGE-2', r2_val, family='eval')
        tf.summary.scalar("ROUGE-L", rl_val, family='eval')
        tf.summary.scalar("R-AVG", r_vag, family='eval')

        tf.summary.scalar('loss_draft', tf.reduce_mean(loss_draft * mask), family='eval')
        tf.summary.scalar('loss_refined', tf.reduce_mean(loss_refined * mask), family='eval')
        tf.summary.scalar("loss", loss, family='eval')

        summaries = tf.summary.merge_all()

        # (batch_size, seq_len), (batch_size, seq_len), scalar, object
        return target_ids[:, :-1], preds_refined_summary, loss, summaries      

    
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


train_iterator = train_dataset.make_initializable_iterator()
train_stream = train_iterator.get_next()

# val_iterator = val_dataset.make_initializable_iterator()
# val_stream = val_iterator.get_next()

xs, ys = train_stream[:3], train_stream[3:]
train_loss, train_op, global_step, train_summaries = model.train(xs, ys)

# xs, ys = val_stream[:3], val_stream[3:]
# y, y_hat, eval_loss, eval_summaries = model.eval(xs, ys)

saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS)


config_tf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config_tf.gpu_options.allocator_type = 'BFC'
config_tf.gpu_options.per_process_gpu_memory_fraction = 0.60
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

    summary_writer_train = tf.summary.FileWriter(os.path.join(config.LOGDIR, 'train'), sess.graph)
    summary_writer_eval = tf.summary.FileWriter(os.path.join(config.LOGDIR, 'eval'), sess.graph)


    initialize_vars(sess)

    _gs = sess.run(global_step)

    sess.run(train_iterator.initializer)
#     sess.run(val_iterator.initializer)

    total_steps = config.NUM_EPOCHS * n_train_batches

    logger.info(f"Running Training Job for '{total_steps}' steps")
    
    for i in tqdm(range(_gs, total_steps+1)):

        _loss, _, _gs, _summary = sess.run([train_loss, train_op, global_step, train_summaries], options=run_options)

        epoch = math.ceil(_gs / n_train_batches)

        summary_writer_train.add_summary(_summary, _gs)
        summary_writer_train.flush() 

#         if (_gs and _gs % n_train_batches == 0):

#             logger.info(f"Epoch '{epoch}' done")
#             logger.info(f"Current training step: '{_gs}")

#             _y, _y_hat, _eval_summary = sess.run([y, y_hat, eval_summaries])

#             summary_writer_eval.add_summary(_eval_summary, 0)
#             summary_writer_eval.flush()       

#             # monitor a random sample
#             rnd = randint(0, _y.shape[0] - 1)

#             y_rnd = ' '.join(tokenizer.convert_ids_to_tokens(_y[rnd]))
#             y_hat_rnd = ' '.join(tokenizer.convert_ids_to_tokens(_y_hat[rnd]))

#             rouges = rouge.get_scores(y_rnd, y_hat_rnd)[0]
#             r1_val, r2_val, rl_val = rouges['rouge-1']["f"], rouges['rouge-2']["f"], rouges['rouge-l']["f"]

#             print('Target:')
#             print(y_rnd)
#             print('Prediction:')
#             print(y_hat_rnd)

#             print(f"ROUGE-1 '{r1_val}'")
#             print(f"ROUGE-2 '{r2_val}'")
#             print(f"ROUGE-L '{rl_val}'")
#             print(f"ROUGE-AVG '{np.mean([r1_val, r2_val, rl_val])}'", '\n--\n')

#             logging.info("Checkpoint: Saving Model")

#             model_output = f"abstractive_summarization_2019_epoch_{epoch}_loss_{str(round(_loss, 4))}"

#             ckpt_name = os.path.join(config.CHECKPOINTDIR, model_output)

#             saver.save(sess, ckpt_name, global_step=_gs)

#             logging.info(f"After training '{_gs}' steps, '{ckpt_name}' has been saved.")

    summary_writer_train.close()  
#     summary_writer_eval.close()  