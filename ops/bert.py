from bert.tokenization import FullTokenizer


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

