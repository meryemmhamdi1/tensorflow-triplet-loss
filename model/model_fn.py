"""Define the model."""

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss


def build_model(is_training, images, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.shape[1:] == [7, 7, num_channels * 2]

    out = tf.reshape(out, [-1, 7 * 7 * num_channels * 2])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


class TripletLoss:
    def __init__(self, params):
        """
        Placeholders for feed_dict:
            images: input batch of images
            labels_images: labels of the images
            tweets: input batch of tweets
            labels_tweets: labels of the tweets
            is_training: bool
            sec_mod: choice of the second modality to be trained using triplet loss against the anchor
            params: contains hyperparameters of the model (ex: `params.learning_rate`)

        """
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.sec_mod = tf.placeholder(dtype=tf.string, name='second_mod')

        self.images = tf.placeholder(shape=(None, 784), dtype=tf.float32, name='input_images')
        self.tweets = tf.placeholder(shape=(None, 768), dtype=tf.float32, name='input_tweets')

        self.labels = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_labels')

        # Dense Layers to unify dimensionalities
        images_dense = tf.keras.layers.Dense(128, activation='relu')(self.images)
        tweets_dense = tf.keras.layers.Dense(128, activation='relu')(self.tweets)

        # -----------------------------------------------------------
        # MODEL: define the layers of the model
        with tf.variable_scope('model'):
            # Compute the embeddings with the model
            embeddings_images = build_model(self.is_training, images_dense, params)
            embeddings_tweets = build_model(self.is_training, tweets_dense, params)

        embeddings = {"image": embeddings_images, "tweet": embeddings_tweets}

        embedding_images_mean_norm = tf.reduce_mean(tf.norm(embeddings_images, axis=1))
        tf.summary.scalar("embedding_images_mean_norm", embedding_images_mean_norm)

        embedding_tweets_mean_norm = tf.reduce_mean(tf.norm(embeddings_tweets, axis=1))
        tf.summary.scalar("embedding_tweets_mean_norm", embedding_tweets_mean_norm)

        if not self.is_training:
            predictions = {'tweets embeddings': embeddings_tweets, "images embeddings": embeddings_images}
            return predictions

        labels = tf.cast(self.labels, tf.int64)

        # Define triplet loss
        if params.triplet_strategy == "batch_all":
            self.loss, fraction = batch_all_triplet_loss(labels, embeddings[self.sec_mod], margin=params.margin,
                                                    squared=params.squared)
        elif params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss(labels, embeddings[self.sec_mod], margin=params.margin,
                                                squared=params.squared)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        # TODO: some other metrics like rank-1 accuracy?
        with tf.variable_scope("metrics"):
            norm_dict = {"image": embedding_images_mean_norm, "tweet": embedding_tweets_mean_norm}
            norm = norm_dict[self.sec_mod]
            self.eval_metric_ops = {"embedding_images_mean_norm": tf.metrics.mean(norm)}

            if params.triplet_strategy == "batch_all":
                self.eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

        # Summaries for training
        tf.summary.scalar('loss', self.loss)
        if params.triplet_strategy == "batch_all":
            tf.summary.scalar('fraction_positive_triplets', fraction)

        # Define training step that minimizes the loss with the Adam optimizer
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        else:
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)


