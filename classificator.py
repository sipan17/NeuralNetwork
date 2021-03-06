import time
import warnings
from os import path, environ

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import (
    load_graph,
    clear_start,
    next_batch,
    freeze_save_graph,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ClassificatorNN:
    """classificatorNN
    Parameters
    ------------
    structure : list (default=[10, 5, 2])
        number of neurons in each layer (including input and output layers)
    activation_fn: function (default tf.nn.relu)
        activation function for hidden layers
    log_dir : string (default='/tmp/log/')
        path to where save the logs of training (tensorboard directory)
    verbose: bool (default=False)
        enable verbose output
    cpu_only: bool (default=True)
        use only cpu
    gpu_fraction : float (default=0.7)
        between (0.0-1.0) how much of the gpu memory allow to use, used if cpu_only is false
    random_state : int
        set random state
    """

    def __init__(self, structure=(10, 5, 2),
                 activation_fn=tf.nn.relu,
                 log_dir='/tmp/log/',
                 verbose=False,
                 cpu_only=True,
                 gpu_fraction=0.7,
                 random_state=None):
        if len(structure) <= 2:
            raise AssertionError('the nerual network should have at least 3 layers: input, hidden, output')
        self.structure = structure
        if activation_fn is None:
            self.activation_fn = tf.identity
        else:
            self.activation_fn = activation_fn
        self._ld = log_dir
        self._verbose = verbose
        if self.structure[-1] < 2:
            raise AssertionError('you should have at least two classes')
        self._cpu_only = cpu_only

        if cpu_only:
            self._config = tf.compat.v1.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
            self._device = "/cpu:0"
        else:
            self._config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            self._config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
            self._device = "/gpu:0"
        self._random_state = None or random_state
        self._input_ph = None
        self._dropout_keep_rate = None
        self._logits = None
        self._output = None
        self._labels = None
        self._sess = None
        self._network = None
        self.ce_losses = None
        self.best_loss = None
        self._encoding = None
        self._loss1 = None
        self._loss2 = None
        self._loss = None
        self._correct_prediction = None
        self._evaluation_step = None
        self._num_batches_train = None
        self._summary_op_step = None
        self._batch_step = None
        self._batch_size_val = None
        self._train_vars = None
        self._summary_op = None
        self._train_op = None
        self._start = None
        self._epoch = None

    def define_loss(self, beta=None):
        self._loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._labels,
                                                                                logits=self._logits), name='ce_loss')
        if beta is None:
            beta = 0

        with tf.name_scope('losses'):
            tf.compat.v1.summary.scalar('cross_entropy_loss', self._loss1)
            self._loss2 = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in self._train_vars]), beta, name='l2_reg_loss')
            self._loss = tf.add(self._loss1, self._loss2, name='total_loss')

            if beta:
                tf.compat.v1.summary.scalar('l2_regularization', self._loss2)
            tf.compat.v1.summary.scalar('total', self._loss)

    def _construct_nn(self, use_batch_norm, seperate_validation):
        tf.compat.v1.reset_default_graph()
        clear_start([self._ld])
        if self._random_state is not None:
            if self._verbose:
                print('seed is fixed to {}'.format(self._random_state))
            tf.compat.v1.set_random_seed(self._random_state)
            np.random.seed(self._random_state)
        layers = []

        self._input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, self.structure[0]], name='input')
        self._dropout_keep_rate = tf.compat.v1.placeholder_with_default(1., shape=None, name='keep_rate')
        self._train_mode = tf.compat.v1.placeholder_with_default(False, shape=None, name='train_mode')
        layers.append(self._input_ph)
        with tf.compat.v1.variable_scope('classifier'):
            for i, n_neurons in enumerate(self.structure[1:-1], 1):
                if i == 1:
                    x = tf.layers.dense(self._input_ph, n_neurons, name='hidden_{}'.format(i),
                                        kernel_initializer=tf.truncated_normal_initializer())
                else:
                    x = tf.layers.dense(x, n_neurons, name='hidden_{}'.format(i),
                                        kernel_initializer=tf.truncated_normal_initializer())
                if use_batch_norm:
                    x = tf.layers.batch_normalization(x, training=self._train_mode, scale=True)
                    layers.append(x)
                x = self.activation_fn(x)
                layers.append(x)
                x = tf.layers.dropout(x, tf.subtract(1., self._dropout_keep_rate), name='hidden_{}'.format(i))
                layers.append(x)
            self._logits = tf.layers.dense(x, self.structure[-1], name='logits',
                                           kernel_initializer=tf.truncated_normal_initializer())
            layers.append(self._logits)
        self._output = tf.nn.softmax(self._logits, name='output')
        layers.append(self._output)

        self._labels = tf.compat.v1.placeholder(tf.float32, shape=[None, self.structure[-1]], name='label')
        layers.append(self._output)
        with tf.device(self._device):
            sess = tf.compat.v1.Session(config=self._config)
        if seperate_validation:
            self._train_writer = tf.compat.v1.summary.FileWriter(path.join(self._ld, 'train'), sess.graph)
            self._val_writer = tf.compat.v1.summary.FileWriter(path.join(self._ld, 'val'))
        else:
            self._train_writer = tf.compat.v1.summary.FileWriter(self._ld, sess.graph)
        self._sess = sess
        self._network = layers

    def fit(self, X, y,
            seperate_validation=True, validation_ratio=0.2,
            learning_rate=0.01, beta=0.0005,
            n_epochs=10, batch_size=16,
            use_batch_norm=True,
            batch_norm_train=True,
            dropout_keep_rate=1.,
            early_stopping_epochs=None,
            early_stopping_method='dlr',
            early_stopping_iters=2,
            save_best_model=False,
            continue_fit=False):

        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param y: array-like, shape (n_samples,)
        Target vector relative to X.
        :param seperate_validation: bool, seperate validation set from X
        :param validation_ratio: float, between (0.0-1.0) the ratio of seperated validation (default=0.2)
        :param learning_rate: float, the learning rate
        :param beta: float, L2 regularization parameter for the weights
        :param use_batch_norm: use batch normalization in hidden layers before activation function
        :param batch_norm_train: bool, train batch_normalization parameters
        :param n_epochs: int, number of epochs to train the network
        :param batch_size: int, batch size
        :param dropout_keep_rate: float, 0.6 would drop 40% of weights
        :param early_stopping_epochs: int, how many epochs to train without improvement (default=None)
        :param early_stopping_method: string, 'dlr' for multiplying learning rate by 0.1 or 'stop'
        :param early_stopping_iters: int, how many time to decrease learning rate (used if 'dlr' is True)
        :param save_best_model: bool, whether or not to save the model with lowest loss
        :param continue_fit, bool, do not reset the graph and continue with current weights
        :return:
        """

        X = np.array(X, np.float32)
        if not continue_fit:
            self._construct_nn(use_batch_norm, seperate_validation)

        if seperate_validation:
            X, val_X, y, val_y = train_test_split(X, y, test_size=validation_ratio,
                                                  stratify=y, random_state=self._random_state)
            assert len(X) > 0, "The training set is empty"
            assert len(val_X) > 0, "The validation set is empty"

        if batch_size is None:
            batch_size = len(X)
        train_inds = np.arange(len(X))
        if not continue_fit:
            self._num_batches_train = len(train_inds) // batch_size
            b_w = 1. / self._num_batches_train
            self._summary_op_step = int(pow(10, np.ceil(np.log10(self._num_batches_train))))
            self._batch_step = int(np.floor(self._summary_op_step * b_w))

        if seperate_validation:
            val_inds = np.arange(len(val_X))
            self._batch_size_val = len(val_inds) // self._num_batches_train

        if not continue_fit:
            train_vars = tf.trainable_variables()
            l2_optimizable_vars = [i for i in train_vars if 'kernel' in i.name.split('/')[-1]]
            self._train_vars = l2_optimizable_vars

            with tf.name_scope('losses'):
                self.define_loss(beta=beta)

            with tf.name_scope('accuracy'):
                self._correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._labels, 1))
                self._evaluation_step = 100 * tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))
                tf.compat.v1.summary.scalar('accuracy', self._evaluation_step)

            self._summary_op = tf.compat.v1.summary.merge_all()
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self._loss)
            self._sess.run(tf.compat.v1.global_variables_initializer())

        if not continue_fit:
            self._start = time.time()
            self.ce_losses = []
            self.best_loss = np.inf

        sleep_time = 0.2
        epochs_not_improved = 0
        decreased_learning_rate = 0

        for epoch in range(n_epochs):
            if continue_fit:
                self._epoch += 1
            else:
                self._epoch = epoch
            # batch_accuracies = []
            if self._verbose:
                print('epoch %d started' % self._epoch)

            q = 0
            cummulative_loss = 0
            for j in range(self._num_batches_train):
                train_inds, batch_inds = next_batch(train_inds, batch_size, shuffle=True)
                batch_features = X[batch_inds]
                batch_labels = y[batch_inds]

                _, train_summary, _loss = self._sess.run([self._train_op, self._summary_op, self._loss],
                                                         feed_dict={self._input_ph: batch_features,
                                                                    self._labels: batch_labels,
                                                                    self._dropout_keep_rate: dropout_keep_rate,
                                                                    self._train_mode: batch_norm_train})

                if seperate_validation:
                    val_inds, batch_inds = next_batch(val_inds, self._batch_size_val)
                    batch_features = val_X[batch_inds]
                    batch_labels = y[batch_inds]
                    assert len(batch_features) > 0, 'empty batch while validation'
                    batch_accuracy, val_summary, _loss = self._sess.run([self._evaluation_step, self._summary_op,
                                                                         self._loss1],
                                                                        feed_dict={self._input_ph: batch_features,
                                                                                   self._labels: batch_labels})

                    self._val_writer.add_summary(val_summary,
                                                 self._epoch * self._summary_op_step + j * self._batch_step)

                cummulative_loss += _loss
                q += 1

                self._train_writer.add_summary(train_summary,
                                               self._epoch * self._summary_op_step + j * self._batch_step)

            mean_loss = cummulative_loss / q
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                if save_best_model:
                    self.save_model(path.join(self._ld, 'best.pb'))
                epochs_not_improved = 0
            else:
                epochs_not_improved += 1

            if self._verbose:
                print('%d epoch mean loss: %f' % (self._epoch, mean_loss))
            self.ce_losses.append(mean_loss)

            if early_stopping_epochs is not None:
                if epochs_not_improved > early_stopping_epochs:
                    if early_stopping_method == 'stop':
                        if self._verbose:
                            print('early stopping')
                        break
                    elif early_stopping_method == 'dlr':
                        if decreased_learning_rate > early_stopping_iters:
                            if self._verbose:
                                print('early stopping')
                            break
                        learning_rate /= 10
                        decreased_learning_rate += 1
                        epochs_not_improved = 0
                        if self._verbose:
                            print('new learning rate {}'.format(learning_rate))
                    else:
                        if self._verbose:
                            print('unknown method for early stopping. stopping the training.')
                        break

            train_inds = np.arange(len(X))
            if seperate_validation:
                val_inds = np.arange(len(val_X))
            time.sleep(sleep_time)
        if self._verbose:
            print('The training took {} seconds'.format(time.time() - self._start - self._epoch * sleep_time))

    def fit_transform(self, X, batch_size=None):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param batch_size: int, batch size
        :return: predicted encodings
        """
        X = np.array(X)
        inds = np.arange(len(X))
        predictions = []
        batch_size = batch_size or len(X)

        start = time.time()
        while len(inds) > 0:
            inds, batch_inds = next_batch(inds, batch_size)

            batch_features = X[batch_inds]
            batch_preds = self._sess.run(self._encoding, feed_dict={self._input_ph: batch_features})
            predictions.extend(batch_preds)

        if self._verbose:
            print('The inference took {} seconds'.format(time.time() - start))
        predictions = np.squeeze(predictions)
        return predictions

    def score(self, X, y, batch_size=None):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
         :param y: array-like, shape (n_samples,)
        Target vector relative to X.
        :param batch_size: int, batch size
        :return: accuracy score
        """
        X = np.array(X)
        y = np.array(y)
        batch_size = batch_size or len(X)
        y_pred = self.predict(X, batch_size)
        return accuracy_score(np.argmax(y, axis=1), y_pred)

    def predict(self, X, batch_size=None):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param batch_size: int, batch size
        :return: predicted class numbers
        """
        predictions = self.predict_proba(X=X, batch_size=batch_size)

        return np.argmax(predictions, -1)

    def predict_proba(self, X, batch_size=None):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param batch_size: int, batch size
        :return: predicted probabilities
        """
        X = np.array(X)
        inds = np.arange(len(X))
        predictions = []
        batch_size = batch_size or len(X)

        start = time.time()
        while len(inds) > 0:
            inds, batch_inds = next_batch(inds, batch_size)

            batch_features = X[batch_inds]
            batch_preds = self._sess.run(self._output, feed_dict={self._input_ph: batch_features})
            predictions.extend(batch_preds)

        if self._verbose:
            print('The inference took {} seconds'.format(time.time() - start))
        predictions = np.squeeze(predictions)
        return predictions

    def save_model(self, path_to_pb):
        freeze_save_graph(self._sess, path.basename(path_to_pb), 'output', path.dirname(path_to_pb))

    def load_model(self, path_to_pb):
        with tf.device(self._device):
            self._input_ph, self._output = load_graph(path_to_pb, ['input:0', 'output:0'])
            self._sess = tf.compat.v1.Session(config=self._config)
        return self
