import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import backend as K


BERT_MODEL_URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"



class BertLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer, integrating BERT from tf-hub
    """
    def __init__(self, url=BERT_MODEL_URL, d_embedding=768, n_fine_tune_layers=0, **kwargs):
        self.url = url
        self.n_fine_tune_layers = n_fine_tune_layers
        self.d_embedding = d_embedding
        
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):   
        
        self.bert = hub.Module(
            self.url,
            trainable=False,
            name="{}_bert_module".format(self.name)
        )
        
        trainable_vars = self.bert.variables
        
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = []

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        
        input_ids, input_mask, segment_ids = inputs
        
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_embedding)

