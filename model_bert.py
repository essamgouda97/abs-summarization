import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

import time
import re
import pickle
import json
import os
 
import math


from decoder import Decoder

class BertModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1, hub_model_name="bert_en_uncased_L-12_H-768_A-12"):
        super(BertModel, self).__init__()

        self.hub_model_name = hub_model_name
 
        self.encoder = hub.KerasLayer(f"https://tfhub.dev/tensorflow/{self.hub_model_name}/3", trainable=True)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, look_ahead_mask, dec_padding_mask):
        # enc_output = self.encoder(inp, training, enc_padding_mask)
        enc_output = self.encoder(inputs=dict(input_word_ids=inp['input_word_ids'][:10],
        input_mask=inp['input_mask'][:10],
        input_type_ids=inp['input_type_ids'][:10],),
        training=False)
        
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights