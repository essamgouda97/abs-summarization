import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
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
from official.nlp import bert
from official.modeling import tf_utils
from official import nlp





import time
import re
import pickle
import os

import wandb
# wandb.init(project="meng-project", tensorboard=True,  sync_tensorboard
# =True)
 
import math
def roundup(x, n):
    return int(math.ceil(x / n) * n)

from custom_lr import CustomSchedule
from model_bert import BertModel

from utils import create_masks
from preprocessing import preprocessing
from preprocessing import bert_encode

#HYPER PARAMS

BUFFER_SIZE = 20000
BATCH_SIZE = 16
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
EPOCHS = 20


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


glue, info = tfds.load('cnn_dailymail', with_info=True, batch_size=-1)

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)


# Set up tokenizer to generate Tensorflow dataset
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)

encoder_vocab_size = len(tokenizer.vocab)
print("Vocab size:", encoder_vocab_size)

# df = pd.read_csv("../datasets/cnn_dailymail.csv")
# document = df['article']
# summary = df['highlights']

glue_train = bert_encode(glue['train'], tokenizer)
for key, value in glue_train.items():
  print(f'{key:15s} shape: {value.shape}')


_, targets, _, decoder_vocab_size, _, summary_tokenizer = preprocessing(None, summary, True)

summary_lengths = pd.Series([len(x) for x in summary])

decoder_maxlen = roundup(summary_lengths.describe(include='all').loc['75%'], 10)

#override
# encoder_maxlen=512
decoder_maxlen=128

#padding/truncatying sequences for identical sequence lengths
# inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')

#shuffle and batch the data using tf's dataset API
# cast to int 32
# inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)


dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(dataset)

#losses adn other metrics

# learning_rate = CustomSchedule(D_MODEL)

# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# train_loss = tf.keras.metrics.Mean(name='train_loss')

# #model
# transformer = BertModel(
#     NUM_LAYERS,
#     D_MODEL,
#     NUM_HEADS,
#     DFF,
#     encoder_vocab_size,
#     decoder_vocab_size,
#     pe_input=encoder_vocab_size,
#     pe_target=decoder_vocab_size,
# )

# #checkpoints
# checkpoint_path = "./checkpoints_cnn_bert"

# ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print("[INFO] Restored latest checkpoint")

# #training
# @tf.function
# def train_step(inp, tar):
#     tar_inp = tar[:, :-1]
#     tar_real = tar[:, 1:]

#     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

#     with tf.GradientTape() as tape:
#         predictions, _ = transformer(
#             inp, tar_inp, 
#             True, 
#             enc_padding_mask, 
#             combined_mask, 
#             dec_padding_mask
#         )
#         loss = loss_function(tar_real, predictions)

#     gradients = tape.gradient(loss, transformer.trainable_variables)    
#     optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

#     train_loss(loss)

# for epoch in range(EPOCHS):
#     start = time.time()
    
#     train_loss.reset_states()
    
#     for (batch, (inp, tar)) in enumerate(dataset):
#         train_step(inp, tar)
        
#         if batch % 429 == 0:
#             print("[EPOCH {}] Batch {} Loss {:.4f}".format(epoch+1, batch, train_loss.result()))
            
#     if (epoch + 1) % 5 == 0: #save every 5 epochs
#         ckpt_save_path = ckpt_manager.save()
#         transformer.save('./models/full_transformer_scratch', overwrite=True) #save entire model
#         print("[INFO] Saving checkpoint for epoch {} at {}".format(epoch+1, ckpt_save_path))

#     print("[Epoch {}] Loss {:.4f}".format(epoch + 1, train_loss.result()))
    
#     print("[INFO] time taken for 1 epoch: {} sec\n".format(time.time() - start))


