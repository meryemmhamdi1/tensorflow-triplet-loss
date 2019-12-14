"""Create the input data pipeline using `tf.data`"""

import model.multi_modal_dataset as multi_modal_dataset
import tensorflow as tf


def train_input_fn(data_dir, params, embed_dir):
    """Train input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = multi_modal_dataset.train(data_dir, embed_dir)
    tf.logging.info("The train dataset was read already...")
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    #tweets = tweets.prefetch(1)  # make sure you always have one batch ready to serve
    dataset = dataset.make_initializable_iterator()

    return dataset


def test_input_fn(data_dir, params, embed_dir):
    """Test input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = multi_modal_dataset.test(data_dir, embed_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.make_initializable_iterator() #prefetch(1) # make sure you always have one batch ready to serve

    return dataset
