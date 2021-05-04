import tensorflow as tf
import tensorflow_datasets as tfds

import pandas as pd


ds, ds_info = tfds.load('cnn_dailymail', split='train', shuffle_files=True, with_info=True)
assert isinstance(ds, tf.data.Dataset)


df = tfds.as_dataframe(ds, ds_info)

df.head()
