import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import backend as K
from tensorflow.python.keras.initializers import Constant

from layers.normalization import LayerNormalization
from ops.encoding import positional_encoding
from ops.attention import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention consists of four parts:
        * Linear layers and split into heads.
        * Scaled dot-product attention.
        * Concatenation of heads.
        * Final linear layer.
    
    Each multi-head attention block gets three inputs;
    Q (query), K (key), V (value).
    These are put through linear (Dense) layers and split up into multiple heads.
    
    Instead of one single attention head, Q, K, and V
    are split into multiple heads because it allows the
    model to jointly attend to information at different
    positions from different representational spaces.
    
    After the split each head has a reduced dimensionality,
    so the total computation cost is the same as a single head
    attention with full dimensionality.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        
    def build(self, input_shape):

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        
        self.dense = tf.keras.layers.Dense(self.d_model)
        
        for i in [self.wq, self.wk, self.wv, self.dense]:
            i.build(input_shape)
            for weight in i.trainable_weights:
                if weight not in self._trainable_weights:
                    self._trainable_weights.append(weight)
            for weight in i.non_trainable_weights:
                if weight not in self._non_trainable_weights:
                    self._non_trainable_weights.append(weight)                    
        
        super(MultiHeadAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights

    
def point_wise_feed_forward_network(d_model, dff):
    """
    Point wise feed forward network consists of two
    fully-connected layers with a ReLU activation in between.
    """
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
    


class EncoderLayer(tf.keras.layers.Layer):
    """
    Each encoder layer consists of sublayers:

        * Multi-head attention (with padding mask)
        * Point wise feed forward networks.
        
    Each of these sublayers has a residual connection around it followed by a layer normalization.
    Residual connections help in avoiding the vanishing gradient problem in deep networks.
    
    The output of each sublayer is `LayerNorm(x + Sublayer(x))`.
    The normalization is done on the d_model (last) axis. There are N encoder layers in the transformer.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate        
        super(EncoderLayer, self).__init__()
        
    def build(self, input_shape):
        
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        
        for i in [self.mha, self.ffn, self.layernorm1, self.layernorm2, self.dropout1, self.dropout2]:
            i.build(input_shape)
            for weight in i.trainable_weights:
                if weight not in self._trainable_weights:
                    self._trainable_weights.append(weight)
            for weight in i.non_trainable_weights:
                if weight not in self._non_trainable_weights:
                    self._non_trainable_weights.append(weight)    
        
        super(EncoderLayer, self).build(input_shape)        

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    Each decoder layer consists of sublayers:

        * Masked multi-head attention (with look ahead mask and padding mask)
        * Multi-head attention (with padding mask). V (value) and K (key)
        receive the encoder output as inputs. Q (query) receives the output from the masked multi-head attention sublayer.
        * Point wise feed forward networks
        
    Each of these sublayers has a residual connection around it followed by a layer normalization.
    The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the d_model (last) axis.
    
    As Q receives the output from decoder's first attention block,
    and K receives the encoder output, the attention weights represent
    the importance given to the decoder's input based on the encoder's output.
    In other words, the decoder predicts the next word by looking at the encoder
    output and self-attending to its own output.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        super(DecoderLayer, self).__init__()
        
    def build(self, input_shape):        

        self.mha1 = MultiHeadAttention(self.d_model, self.num_heads)
        self.mha2 = MultiHeadAttention(self.d_model, self.num_heads)

        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.dropout3 = tf.keras.layers.Dropout(self.rate)
        
        for i in [self.mha1, self.mha2, self.ffn, self.layernorm1, self.layernorm2, self.layernorm3, self.dropout1, self.dropout2, self.dropout3]:
            i.build(input_shape)
            for weight in i.trainable_weights:
                if weight not in self._trainable_weights:
                    self._trainable_weights.append(weight)
            for weight in i.non_trainable_weights:
                if weight not in self._non_trainable_weights:
                    self._non_trainable_weights.append(weight)    
        
        super(DecoderLayer, self).build(input_shape)            


    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """
    The Encoder consists of:
        
        1. Input Embedding
        2. Positional Encoding
        3. N encoder layers
        
    The input is put through an embedding which is summed with the positional encoding.
    The output of this summation is the input to the encoder layers.
    The output of the encoder is the input to the decoder.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.rate = rate
        
    def build(self, input_shape):
        
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(self.input_vocab_size, self.d_model)


        self.enc_layers = [
            EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
            for _ in range(self.num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(self.rate)
        
        for i in [self.embedding] + self.enc_layers + [self.dropout]:
            i.build(input_shape)
            for weight in i.trainable_weights:
                if weight not in self._trainable_weights:
                    self._trainable_weights.append(weight)
            for weight in i.non_trainable_weights:
                if weight not in self._non_trainable_weights:
                    self._non_trainable_weights.append(weight)    
        
        super(Encoder, self).build(input_shape)                    

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """
    The Decoder consists of:
        1. Output Embedding
        2. Positional Encoding
        3. N decoder layers

    The target is put through an embedding which is summed with the positional encoding.
    The output of this summation is the input to the decoder layers.
    The output of the decoder is the input to the final linear layer.
    
    If `pretrained_embeddings` are available, we use it as a word embedding matrix and do not perform further training
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.rate = rate        

#         if pretrained_embeddings is None:
#             self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
#         else:
#             self.embedding = tf.keras.layers.Embedding(
#                 target_vocab_size, d_model, trainable=False,
#                 embeddings_initializer=Constant(pretrained_embeddings)
#             )

    def build(self, input_shape):
            
        self.pos_encoding = positional_encoding(self.target_vocab_size, self.d_model)

        self.dec_layers = [
            DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate) 
            for _ in range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.rate)
        
        for i in self.dec_layers + [self.dropout]:
            i.build(input_shape)
            for weight in i.trainable_weights:
                if weight not in self._trainable_weights:
                    self._trainable_weights.append(weight)
            for weight in i.non_trainable_weights:
                if weight not in self._non_trainable_weights:
                    self._non_trainable_weights.append(weight)    
        
        super(Decoder, self).build(input_shape)          

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        
        seq_len = tf.shape(x)[1]
        attention_weights = {}

#         if not input_alreay_embedded:
#             x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
            
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            
#             dv = f"/device:GPU:{str(next(selector))}"
#             print(f"With device )
#             with tf.device():
                
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights



