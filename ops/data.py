import re
import time
import glob
import struct
import shutil
import logging
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from collections import namedtuple
from itertools import count
from functools import partial
from functools import reduce
from functools import wraps

from tensorflow.core.example import example_pb2
from bert.tokenization import FullTokenizer
from tqdm import tqdm

from utils.decorators import timeit
from config import config


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()



SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

BERT_MODEL_URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

        
InputExample = namedtuple('InputExample', ['guid', 'text_a', 'text_b'])

InputFeatures = namedtuple('InputFeatures', ['guid', 'tokens', 'input_ids', 'input_mask', 'input_type_ids'])

        
def pad(l, n, pad):
    """
    Pad the list 'l' to have size 'n' using 'padding_element'
    """
    return l + [pad] * (n - len(l))


def calc_num_batches(total_num, batch_size):
    """
    Calculates the number of batches, allowing for remainders.
    """
    return total_num // batch_size + int(total_num % batch_size != 0)
           

def convert_single_example(tokenizer, example, max_seq_len=config.SEQ_LEN):
    """
    Convert `text` to the Bert input format
    """
    tokens = tokenizer.tokenize(example.text_a)
    
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[0:(max_seq_len - 2)]
        
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_type_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    
    input_ids = pad(input_ids, max_seq_len, 0)
    input_type_ids = pad(input_type_ids, max_seq_len, 0)
    input_mask = pad(input_mask, max_seq_len, 0)
        
    return tokens, input_ids, input_mask, input_type_ids
    
    
def convert_examples_to_features(tokenizer, examples, max_seq_len=config.SEQ_LEN):
    """
    Convert raw features to Bert specific representation
    """
    converter = partial(convert_single_example, tokenizer=tokenizer, max_seq_len=max_seq_len)
    examples = [converter(example=example) for example in examples]
    return examples


def create_tokenizer_from_hub_module(bert_hub_url):
    """
    Get the vocab file and casing info from the Hub module.
    """
    bert_module = hub.Module(bert_hub_url)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)

    with tf.Session() as sess:
        vocab_file, do_lower_case = sess.run([
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"]
        ])
        
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def abstract2sents(abstract):
    """
    # Use the <s> and </s> tags in abstract to get a list of sentences.
    """
    sentences_pattern = re.compile(r"<s>(.+?)<\/s>")
    sentences = sentences_pattern.findall(abstract)
    return sentences


def _load_single(file):
    """
    Opens and prepares a single chunked file
    """
    article_texts = []
    abstract_texts = []
    
    with open(file, 'rb') as f:
    
        while True:
            
            len_bytes = f.read(8)

            if not len_bytes:
                break

            str_len = struct.unpack('q', len_bytes)[0]
            str_bytes = struct.unpack('%ds' % str_len, f.read(str_len))[0]
            example = example_pb2.Example.FromString(str_bytes) 

            article_text = example.features.feature['article'].bytes_list.value[0].decode('unicode_escape').strip()
            abstract_text = example.features.feature['abstract'].bytes_list.value[0].decode('unicode_escape').strip()
            abstract_text = ' '.join([sent.strip() for sent in abstract2sents(abstract_text)])

            article_texts.append(article_text)
            abstract_texts.append(abstract_text)
        
    return article_texts, abstract_texts


def load_data(files):
    """
    Reads binary data and returns chuncks of [(articles, summaries)]
    """
    logger.info(f"'{len(files)}' files found")
    data = [_load_single(file) for file in files]
    articles = sum([a for a, _ in data], [])
    summaries = sum([b for _, b in data], [])
    return articles, summaries


def convert_single_example(example, tokenizer, max_seq_len=MAX_SEQ_LEN):
    """
    Convert `text` to the Bert input format
    """
    tokens = tokenizer.tokenize(example)
    
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[0:(max_seq_len - 2)]
        
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_type_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    
    input_ids = pad(input_ids, max_seq_len, 0)
    input_type_ids = pad(input_type_ids, max_seq_len, 0)
    input_mask = pad(input_mask, max_seq_len, 0)
        
    return tokens, input_ids, input_mask, input_type_ids


def generator_fn(sents1, sents2, tokenizer):
    """
    Generates training / evaluation data
    raw_data: list of (abstracts, raw_data)
    tokenizer: tokenizer to separate the different tokens
    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    ys: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    """    
    for article, summary in zip(sents1, sents2):
        tokens_x, input_ids_x, input_mask_x, input_type_ids_x = convert_single_example(article, tokenizer)
        tokens_y, input_ids_y, input_mask_y, input_type_ids_y = convert_single_example(summary, tokenizer)
        
        x_seqlen, y_seqlen = len(tokens_x), len(tokens_y)
        
        yield (input_ids_x, input_mask_x, input_type_ids_x, x_seqlen, article), (input_ids_y, input_mask_y, input_type_ids_y, y_seqlen, summary)

    
    
def input_fn(sents1, sents2, tokenizer, batch_size, shuffle=False):
    """
    Batchify data
    raw_data [(artciles, abstracts)]
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    """
    shapes = (
        ([None], [None], [None], (), ()),
        ([None], [None], [None], (), ())
    )
    
    types = (
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.string),
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.string)
    )
    
    paddings = (
        (0, 0, 0, 0, ''),
        (0, 0, 0, 0, '')
    )

    dataset = tf.data.Dataset.from_generator(
        partial(generator_fn, tokenizer=tokenizer),
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset    
    
    
@timeit
def prepare_data(inputpath, batch_size, shuffle=False):
    """
    """
    files = glob.glob(inputpath + '*.bin')[:2]
                
    sents1, sents2 = load_data(files)
    
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_URL)        

    batches = input_fn(sents1, sents2, tokenizer, batch_size, shuffle=shuffle)
    
    num_batches = calc_num_batches(len(sents1), batch_size)
    
    return batches, num_batches, len(sents1)    
    
     
