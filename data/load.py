import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from functools import partial
from tensorflow.keras import backend as K
from ops.tokenization import tokenizer

from config import config


# Special Tokens
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103


def pad(l, n, pad=0):
    """
    Pad the list 'l' to have size 'n' using 'padding_element'
    """
    pad_with = (0, max(0, n - len(l)))
    return np.pad(l, pad_with, mode='constant', constant_values=pad)


def encode(sent_1, sent_2, tokenizer, input_seq_len, output_seq_len):
    """
    Encode the text to the BERT expected format
    
    'input_seq_len' is used to truncate the the article length
    'output_seq_len' is used to truncate the the summary length

    BERT has the following special tokens:    
    
    [CLS] : The first token of every sequence. A classification token
    which is normally used in conjunction with a softmax layer for classification
    tasks. For anything else, it can be safely ignored.

    [SEP] : A sequence delimiter token which was used at pre-training for
    sequence-pair tasks (i.e. Next sentence prediction). Must be used when
    sequence pair tasks are required. When a single sequence is used it is just appended at the end.

    [MASK] : Token used for masked words. Only used for pre-training.
    
    Additionally BERT requires additional inputs to work correctly:
        - Mask IDs
        - Segment IDs
    
    The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    Sentence Embeddings is just a numeric class to distinguish between pairs of sentences.
    """
    tokens_1 = tokenizer.tokenize(sent_1.numpy())
    tokens_2 = tokenizer.tokenize(sent_2.numpy())
    
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_1) > input_seq_len - 2:
        tokens_1 = tokens_1[0:(input_seq_len - 2)]
    if len(tokens_2) > (output_seq_len + 1) - 2:
        tokens_2 = tokens_2[0:((output_seq_len + 1) - 2)]
        
    tokens_1 = ["[CLS]"] + tokens_1 + ["[SEP]"]
    tokens_2 = ["[CLS]"] + tokens_2 + ["[SEP]"]
    
    input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
    input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
    
    input_mask_1 = [1] * len(input_ids_1)
    input_mask_2 = [1] * len(input_ids_2)

    input_ids_1 = pad(input_ids_1, input_seq_len, 0)
    input_ids_2 = pad(input_ids_2, output_seq_len + 1, 0)
    input_mask_1 = pad(input_mask_1, input_seq_len, 0)
    input_mask_2 = pad(input_mask_2, output_seq_len + 1, 0)
    
    input_type_ids_1 = [0] * len(input_ids_1)
    input_type_ids_2 = [0] * len(input_ids_2)
    
    return input_ids_1, input_mask_1, input_type_ids_1, input_ids_2, input_mask_2, input_type_ids_2


def tf_encode(tokenizer, input_seq_len, output_seq_len):
    """
    Operations inside `.map()` run in graph mode and receive a graph
    tensor that do not have a `numpy` attribute.
    The tokenizer expects a string or Unicode symbol to encode it into integers.
    Hence, you need to run the encoding inside a `tf.py_function`,
    which receives an eager tensor having a numpy attribute that contains the string value.
    """    
    def f(s1, s2):
        encode_ = partial(encode, tokenizer=tokenizer, input_seq_len=input_seq_len, output_seq_len=output_seq_len)
        return tf.py_function(encode_, [s1, s2], [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])
    
    return f


def filter_max_length(x, x1, x2, y, y1, y2, max_length=config.MAX_EXAMPLE_LEN):
    predicate = tf.logical_and(
        tf.size(x[0]) <= max_length,
        tf.size(y[0]) <= max_length
    )
    return predicate


def pipeline(examples, tokenizer, cache=False):
    """
    Prepare a Dataset to return the following elements
    x_ids, x_mask, x_segments, y_ids, y_maks, y_segments
    """
    
    dataset = examples.map(tf_encode(tokenizer, config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if config.MAX_EXAMPLE_LEN is not None:
        dataset = dataset.filter(filter_max_length)

    if cache:
        dataset = dataset.cache()
        
    dataset = dataset.shuffle(config.BUFFER_SIZE).padded_batch(config.BATCH_SIZE, padded_shapes=([-1], [-1], [-1], [-1], [-1], [-1]))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset


def load_cnn_dailymail(tokenizer=tokenizer):
    """
    Load the CNN/DM data from tensorflow Datasets
    """
    examples, metadata = tfds.load('cnn_dailymail', with_info=True, as_supervised=True)
    train, val, test = examples['train'], examples['validation'], examples['test']
    train_dataset = pipeline(train, tokenizer)
    val_dataset = pipeline(val, tokenizer)
    test_dataset = pipeline(test, tokenizer)
    
    # grab information regarding the number of examples
    metadata = json.loads(metadata.as_json)

    n_test_examples = int(metadata['splits'][0]['statistics']['numExamples'])
    n_train_examples = int(metadata['splits'][1]['statistics']['numExamples'])
    n_val_examples = int(metadata['splits'][2]['statistics']['numExamples'])
    
    return train_dataset, val_dataset, test_dataset, n_train_examples, n_val_examples, n_test_examples
    
