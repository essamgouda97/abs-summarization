import tensorflow as tf
import pandas as pd
import numpy as np

from model import Transformer
from preprocessing import preprocessing
from utils import create_masks
from custom_lr import CustomSchedule

import math
def roundup(x, n):
    return int(math.ceil(x / n) * n)

def evaluate(input_document, summary_tokenizer, document_tokenizer):
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [summary_tokenizer.word_index["<start>"]]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["<end>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def summarize(input_document, summary_tokenizer, document_tokenizer):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0, summary_tokenizer, document_tokenizer)  # not printing <s> token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document

#HYPER PARAMS

BUFFER_SIZE = 20000
BATCH_SIZE = 16
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
EPOCHS = 20


df = pd.read_csv("../datasets/cnn_dailymail.csv")
document = df['article']
summary = df['highlights']

inputs, targets, encoder_vocab_size, decoder_vocab_size, document_tokenizer, summary_tokenizer = preprocessing(document, summary)

document_lengths = pd.Series([len(x) for x in document])
summary_lengths = pd.Series([len(x) for x in summary])

# taking the 75th percentile as the cutting point and round it and not leaving high variance
encoder_maxlen = roundup(document_lengths.describe(include='all').loc['75%'], 100)
decoder_maxlen = roundup(summary_lengths.describe(include='all').loc['75%'], 10)

#override
encoder_maxlen=512
decoder_maxlen=128

#model
transformer = Transformer(
    NUM_LAYERS,
    D_MODEL,
    NUM_HEADS,
    DFF,
    encoder_vocab_size,
    decoder_vocab_size,
    pe_input=encoder_vocab_size,
    pe_target=decoder_vocab_size,
)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

#checkpoints
checkpoint_path = "./checkpoints"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("[INFO] Restored latest checkpoint")


test_text = "US-based private equity firm General Atlantic is in talks to invest about \
    $850 million to $950 million in Reliance Industries' digital unit Jio \
    Platforms, the Bloomberg reported. Saudi Arabia's $320 billion sovereign \
    wealth fund is reportedly also exploring a potential investment in the \
    Mukesh Ambani-led company. The 'Public Investment Fund' is looking to \
    acquire a minority stake in Jio Platforms."
print(summarize(test_text, summary_tokenizer))


from rouge import Rouge

hypothesis = summarize(document[30], summary_tokenizer)
reference = summary[30]
rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)

