#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gzip
import os
import shutil

import numpy as np
from six.moves import urllib
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import requests


# Find the maximum length of any tweet in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def download_image(url):
    img_data = requests.get(url).content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)


def decode_image_with_padding(im_file, decode_fn=tf.image.decode_jpeg, channels=3, pad_upto=500):
    im_orig = decode_fn(tf.io.read_file(im_file), channels=channels)
    img_shape = tf.shape(im_orig)[0:2]

    pad_h = pad_upto - img_shape[0]
    pad_w = pad_upto - img_shape[1]

    if channels == 3:
        padded_R = tf.pad(tensor=im_orig[:, :, 0], #the R channel of tensor
                          paddings=[[0, pad_h], [0, pad_w]],
                          mode='Constant',
                          name='padAverageRed',
                          constant_values=0)

        padded_G = tf.pad(tensor=im_orig[:, :, 1], #the G channel of tensor
                          paddings=[[0, pad_h], [0, pad_w]],
                          mode='Constant',
                          name='padAverageGreen',
                          constant_values=0)

        padded_B = tf.pad(tensor=im_orig[:, :, 2], #the B channel of tensor
                          paddings=[[0, pad_h], [0, pad_w]],
                          mode='Constant',
                          name='padAverageBlue',
                          constant_values=0)

        img_padded = tf.stack([padded_R, padded_G, padded_B], axis=2)

    else:
        padded_R = tf.pad(tensor=im_orig[:, :, 0], #the R channel of tensor
                          paddings=[[0, pad_h], [0, pad_w]],
                          mode='Constant',
                          name='padAverageRed',
                          constant_values=0)

        img_padded = tf.stack([padded_R], axis=2)

    return img_shape, img_padded


def preprocess_tweets(tweets):
    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(tweets)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(tweets)

    # Pad each vector to the max_length of the tweets
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    tweet_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    return tweet_vector, max_length


def dataset(directory, csv_file_path, anchor):
    """ Read dataset which consists of tweets, images, user raw data or embeddings and their corresponding
     labels "user_id" """

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(directory + image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    data = pd.read_cscv(os.path.join(directory, csv_file_path))

    images = tf.data.Dataset.from_tensor_slices(list(data["image_paths"])).map(decode_image)
    labels = tf.data.Dataset.from_tensor_slices(list(data["user_id"])).map(decode_label)

    tweets = preprocess_tweets(list(data["text"]))

    return tf.data.Dataset.zip((images, labels)), tf.data.Dataset.zip((tweets, labels))


def train(directory, anchor):
    """tf.data.Dataset object for multi-modal training data."""
    return dataset(directory, 'dota_col_red_train.csv', anchor)


def test(directory, anchor):
    """tf.data.Dataset object for multi-modal test data."""
    return dataset(directory, 'test_col_red_test.csv', anchor)
