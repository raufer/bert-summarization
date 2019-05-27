import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import backend as K
from layers.transformer import Encoder
from layers.transformer import Decoder


class Transformer(tf.keras.Model):
    """
    Transformer consists of the encoder, decoder and a final linear layer.
    The output of the decoder is the input to the linear layer and its output is returned.    
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
