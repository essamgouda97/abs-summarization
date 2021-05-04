import tensorflow as tf
import numpy as np

def preprocessing(document, summary, bert=False):
    # for decoder sequence
    summary = summary.apply(lambda x: '<start> ' + x + ' <end>')

    #tokenizing
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
    oov_token = '<unk>'
    if not bert:
        document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
        document_tokenizer.fit_on_texts(document)
        inputs = document_tokenizer.texts_to_sequences(document)
        encoder_vocab_size = len(document_tokenizer.word_index) + 1
    else:
        inputs = None
        encoder_vocab_size = None
        document_tokenizer = None
    
    summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
    summary_tokenizer.fit_on_texts(summary)

    targets = summary_tokenizer.texts_to_sequences(summary)
    
    decoder_vocab_size = len(summary_tokenizer.word_index) + 1

    return inputs, targets, encoder_vocab_size, decoder_vocab_size, document_tokenizer, summary_tokenizer

def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(glue_dict, tokenizer):


    article = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict)])
    # highlights = tf.ragged.constant([
    #     encode_sentence(s, tokenizer)
    #     for s in np.array(glue_dict["highlights"])])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*article.shape[0]
    input_word_ids = tf.concat([cls, article], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(article)
    # type_s2 = tf.ones_like(highlights)
    input_type_ids = tf.concat(
        [type_cls, type_s1], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs