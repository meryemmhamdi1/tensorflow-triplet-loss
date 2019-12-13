"""Train the model"""

import argparse
import os

import tensorflow as tf

from model.input_fn import train_input_fn
from model.input_fn import test_input_fn
from model.model_fn import TripletLoss
from model.utils import Params
import random


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='/Users/d22admin/USCGDrive/BeyondAssignment/IntermediateData/',
                    help="Directory containing the dataset")
parser.add_argument("--bert_model", default="bert-base-cased")


ANCHORS = ["tweet", "image"]
if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Get the datasets
    tf.logging.info("Getting the datast...")
    dataset_iter = train_input_fn(args.data_dir, params, args.bert_model)
    dataset_next = dataset_iter.get_next()

    # Define the model
    tf.logging.info("Creating the model...")
    model = TripletLoss()

    num_train = 5031
    num_train_steps = int(num_train/ params.batch_size) * params.num_epochs
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(dataset_next.initializer)

        """ Training Module """
        for i in tqdm(range(0, num_train_steps)):
            train = sess.run(dataset_next)

            fd_train = {
                model.is_training: True,
                model.sec_mod: random.choice(ANCHORS),
                model.images: dataset_next[0],
                model.tweets: dataset_next[1],
                model.labels: dataset_next[2]
            }

            _, train_loss, train_acc = sess.run([model.train_optimizer, model.total_loss, model.acc], fd_train)
            print("iteration:", i, " train_loss:", train_loss, " train_acc:", train_acc)

