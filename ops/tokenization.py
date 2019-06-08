import tensorflow as tf
import tensorflow_hub as hub

from bert.tokenization import FullTokenizer


BERT_MODEL_URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def create_bert_tokenizer(vocab_file, do_lower_case=True):
    """
    Return a BERT FullTokenizer
    """                
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def create_tokenizer_from_hub_module(bert_hub_url, bert_module=None):
    """
    Get the vocab file and casing info from the Hub module.
    """
    if bert_module is None:
        bert_module = hub.Module(bert_hub_url)
        
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)

    with tf.Session() as sess:
        vocab_file, do_lower_case = sess.run([
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"]
        ])
                
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_URL)


def convert_idx_to_token_tensor(inputs, tokenizer=tokenizer):
    '''
    Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    tokenizer :: [int] -> str
    Returns
    1d string tensor.
    '''
    def f(inputs):
        return ' '.join(tokenizer.convert_ids_to_tokens(inputs))

    return tf.py_func(f, [inputs], tf.string)
