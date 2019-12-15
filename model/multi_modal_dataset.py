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
import requests
from tqdm import tqdm
import ast

# Find the maximum length of any tweet in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def download_image(url):
    img_data = requests.get(url).content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)


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
    tweet_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post', maxlen=50)

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    return tweet_vector, max_length


def dataset(directory, csv_file_path, embed_dir, text_embed_file, img_embed_file):
    """ Read dataset which consists of tweets, images, user raw data or embeddings and their corresponding
     labels "user_id" """

    def decode_image(image_path):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize_images(image, size=[300, 300], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        return image

    def get_img_embeddings(index):
        # Normalize from [0, 255] to [0.0, 1.0]
        img_emb = np.load(embed_dir+img_embed_file)["arr_0"][index]
        return img_emb

    def decode_label(label):
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    def decode_team(label):
        return tf.one_hot(label, 16)

    """
    def process_bert_tokenize(text):

        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Mark each of the 22 tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, pooled_output = model(tokens_tensor, segments_tensors)

        # Get sentence embeddings from CLS Token
        sentence_embeddings = tf.convert_to_tensor(np.array(np.squeeze(pooled_output.numpy())), dtype=tf.float32)
        return sentence_embeddings

    """

    def get_bert_embeddings(index):
        text_emb = np.load(embed_dir+text_embed_file)["arr_0"][index]
        return text_emb

    def get_index_one(l):
        l = ast.literal_eval(l)
        index = list(map(lambda i: i > 0.6, l)).index(True)
        return index

    def list_to_dict(l):
        op = {l[i]: i for i in range(0, len(l))}
        return op

    tf.logging.info("Reading the dataset...")
    data = pd.read_csv(os.path.join(directory, csv_file_path))
    data["index"] = list(data.index)

    tf.logging.info("Reading the tweets...")
    tweets_embed = np.array([get_bert_embeddings(data["index"].iloc[i]) for i in range(0, len(data))])
    tweets = tf.data.Dataset.from_tensor_slices(tweets_embed)

    tf.logging.info("Reading the images...")
    img_embed = np.array([get_img_embeddings(data["index"].iloc[i]) for i in range(0, len(data))])
    images = tf.data.Dataset.from_tensor_slices(img_embed)

    tf.logging.info("Reading the labels...")
    user_list = list(set(list(data["user_id"])))
    user_dict = list_to_dict(user_list)
    user_list_small = [user_dict[el] for el in user_list]
    user_ids = tf.data.Dataset.from_tensor_slices(user_list_small).map(decode_label)

    tf.logging.info("Reading the teams one hot encoding...")
    teams = tf.data.Dataset.from_tensor_slices([get_index_one(el) for el in list(data['teams_one_hot'])])

    tf.logging.info("Finished reading tweets embeddings :)")

    return tf.data.Dataset.zip((images, tweets, user_ids, teams))


def train(directory, embed_dir):
    """tf.data.Dataset object for multi-modal training data."""
    return dataset(directory, 'df_tripets_with_teams_2_4k_train_useful.csv', embed_dir,
                   "text_emb_train_2k_bert_base_cased_useful.npz", "img_emb_2k_img2vec_train.npz")


def test(directory, embed_dir):
    """tf.data.Dataset object for multi-modal test data."""
    return dataset(directory, 'df_tripets_with_teams_2_4k_test_useful.csv', embed_dir,
                   "text_emb_test_2k_bert_base_cased_useful.npz", "img_emb_2k_img2vec_test.npz")
