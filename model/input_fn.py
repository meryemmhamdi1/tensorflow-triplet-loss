"""Create the input data pipeline using `tf.data`"""

import model.multi_modal_dataset as multi_modal_dataset


def train_input_fn(data_dir, params, bert_model):
    """Train input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = multi_modal_dataset.train(data_dir, bert_model)
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    #tweets = tweets.prefetch(1)  # make sure you always have one batch ready to serve
    dataset = dataset.make_initializable_iterator()

    return dataset


def test_input_fn(data_dir, params, bert_model):
    """Test input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = multi_modal_dataset.test(data_dir, bert_model)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.make_initializable_iterator() #prefetch(1) # make sure you always have one batch ready to serve

    return dataset
