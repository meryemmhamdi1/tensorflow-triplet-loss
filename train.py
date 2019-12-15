"""Train the model"""

import argparse
import os

import tensorflow as tf

from model.input_fn import train_input_fn
from model.input_fn import test_input_fn
from model.model_fn import TripletLoss
from model.utils import Params
import random
from tqdm import tqdm
from numpy import savez_compressed
import model.multi_modal_dataset as multi_modal_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/batch_hard',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='/Users/d22admin/USCGDrive/BeyondAssignment/small_dataset_useful/',
                    help="Directory containing the dataset")

parser.add_argument('--embed_dir', default='/Users/d22admin/USCGDrive/BeyondAssignment/Deliverables/Embeddings/')
parser.add_argument("--bert_model", default="bert-base-cased")
parser.add_argument("--result_dir", default="/Users/d22admin/USCGDrive/BeyondAssignment/Deliverables/Results/")


ANCHORS = [0, 1, 2] # 0 for "tweet", 1 for "image", 2 for "user"
if __name__ == '__main__':
    tf.reset_default_graph()
    #tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Get the datasets
    #tf.logging.info("Getting the dataset...")
    dataset_iter = train_input_fn(args.data_dir, params, args.embed_dir)
    dataset_next = dataset_iter.get_next()

    # Define the model
    #tf.logging.info("Creating the model...")
    model = TripletLoss(params)

    num_train = 5031
    num_train_steps = int(num_train/params.batch_size) * params.num_epochs
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        """ Training Module """
        for i in tqdm(range(0, num_train_steps)):
            sess.run(dataset_iter.initializer)
            train = sess.run(dataset_next)

            fd_train = {
                model.is_training: True,
                model.anchor_mode: random.choice(ANCHORS),
                model.opt_mode: random.choice(ANCHORS),
                model.images: train[0],
                model.tweets: train[1],
                model.user_ids: train[2],
                model.team: train[3]
            }

            _, train_loss, train_acc, tweets_dense, images_dense = sess.run([model.train_op, model.total_loss,
                                                                             model.class_acc, model.tweets_dense,
                                                                             model.images_dense], fd_train)
            print("iteration:", i, " train_loss:", train_loss, " train_acc:", train_acc)

        tf.logging.info("Finished training!! Now saving model and embeddings")
        model.save(os.path.join(args.model_dir, "sample_multi_modal_model"), sess)

        tf.logging.info("Visualizing the embeddings on the TRAIN portion!")

        dataset = multi_modal_dataset.train(args.data_dir, args.embed_dir)
        dataset = dataset.batch(1684)
        dataset = dataset.prefetch(1)
        dataset = dataset.make_initializable_iterator()
        dataset_next = dataset.get_next()

        # Obtain the test labels
        dataset_next = dataset.get_next()

        sess.run(dataset.initializer)
        train = sess.run(dataset_next)

        fd_train = {
            model.is_training: False,
            model.anchor_mode: random.choice(ANCHORS),
            model.opt_mode: random.choice(ANCHORS),
            model.images: train[0],
            model.tweets: train[1],
            model.user_ids: train[2],
            model.team: train[3]
        }

        train_acc, tweets_dense, images_dense = sess.run([model.class_acc, model.tweets_dense, model.images_dense],
                                                         fd_train)
        print("tweets_dense.shape:", tweets_dense.shape, " images_dense.shape:", images_dense.shape, "train acc:",
              train_acc)

        savez_compressed(os.path.join(args.result_dir, 'text_emb_sample_both_train.npz'), tweets_dense)

        savez_compressed(os.path.join(args.result_dir, 'img_emb_sample_both_train.npz'), images_dense)

        tf.logging.info("Visualizing the embeddings on the TEST portion!")
        dataset = multi_modal_dataset.test(args.data_dir, args.embed_dir)
        dataset = dataset.batch(421)
        dataset = dataset.prefetch(1)
        dataset = dataset.make_initializable_iterator()
        dataset_next = dataset.get_next()

        # Obtain the test labels
        dataset_next = dataset.get_next()

        sess.run(dataset.initializer)
        test = sess.run(dataset_next)

        fd_test = {
            model.is_training: False,
            model.anchor_mode: random.choice(ANCHORS),
            model.opt_mode: random.choice(ANCHORS),
            model.images: test[0],
            model.tweets: test[1],
            model.user_ids: test[2],
            model.team: test[3]
        }

        test_acc, tweets_dense, images_dense = sess.run([model.class_acc, model.tweets_dense, model.images_dense], fd_test)
        print("tweets_dense.shape:", tweets_dense.shape, " images_dense.shape:", images_dense.shape, "test acc:",
              test_acc)

        savez_compressed(os.path.join(args.result_dir, 'text_emb_sample_both_test.npz'), tweets_dense)

        savez_compressed(os.path.join(args.result_dir, 'img_emb_sample_both_test.npz'), images_dense)








