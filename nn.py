from os import path
import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
from tensorflow.python.framework import ops
from utils import clear_start, next_batch, next_batch_shuffle, freeze_save_graph, load_graph
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time


class NeuralNetwork:
    """NeuralNetwork
    Parameters
    ------------
    structure : list (default=[10, 5, 1])
        number of neurons in each layer (including input and output layers)
    verbose: bool (default=False)
        enable verbose output
    cpu_only: bool (default=True)
        use only cpu
    cpu_number: int (default=0)
        which cpu to use
    gpu_fraction : float (default=0.9)
        between (0.0-1.0) how much of the gpu memory allow to use, used if cpu_only is false
    gpu_number: int (default=0)
        which gpu to use
    random_state : int
        set random state
    log_dir : string (default='/tmp/log/')
        path to where save the logs of training (tensorboard directory)
    """

    def __init__(self, structure=[10, 5, 1],
                 activation_fn=tf.nn.relu,
                 log_dir='/tmp/log/',
                 verbose=False,
                 cpu_only=True, cpu_number=0,
                 gpu_fraction=0.9, gpu_number=0,
                 random_state=None):
        assert len(structure) > 2, 'the nerual network should have at least 3 layers: input, hidden, output'
        self.structure = structure
        self.activation_fn = activation_fn
        self._ld = log_dir
        self._verbose = verbose
        self._is_binary = True if self.structure[-1] == 1 else False
        self._cpu_only = cpu_only

        if cpu_only:
            self._config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
            self._cpu_number = cpu_number
        else:
            self._gpu_number = gpu_number
            self._config = tf.ConfigProto(allow_soft_placement=True)
            self._config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
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

    def _construct_nn(self, seperate_validation):
        ops.reset_default_graph()
        clear_start([self._ld])
        if self._random_state:
            tf.set_random_seed(self._random_state)
            np.random.seed(self._random_state)
        layers = []

        self._input_ph = tf.placeholder(tf.float32, shape=[None, self.structure[0]], name='input')
        self._dropout_keep_rate = tf.placeholder_with_default(1., shape=None, name='keep_rate')
        layers.append(self._input_ph)
        for i, n_neurons in enumerate(self.structure[1:-1]):
            if i == 0:
                x = tf.layers.dense(self._input_ph, n_neurons, activation=self.activation_fn, name='hidden_%s' % (i + 1),
                                    kernel_initializer=tf.truncated_normal_initializer())
            else:
                x = tf.layers.dense(x, n_neurons, activation=self.activation_fn, name='hidden_%s' % (i + 1),
                                    kernel_initializer=tf.truncated_normal_initializer())
            layers.append(x)
            x = tf.layers.dropout(x, tf.subtract(1., self._dropout_keep_rate), name='dropout_%s' % (i + 1))
        self._logits = tf.layers.dense(x, self.structure[-1], name='logits',
                                       kernel_initializer=tf.truncated_normal_initializer())
        if self._is_binary:
            self._output = tf.nn.sigmoid(self._logits, name='output')
        else:
            self._output = tf.nn.softmax(self._logits, name='output')
        self._labels = tf.placeholder(tf.float32, shape=[None, self.structure[-1]], name='label')
        layers.append(self._logits)
        if self._cpu_only:
            with tf.device('/cpu:{}'.format(self._cpu_number)):
                with tf.Session(config=self._config) as sess:
                    if seperate_validation:
                        self._train_writer = tf.summary.FileWriter(self._ld + 'train/', sess.graph)
                        self._val_writer = tf.summary.FileWriter(self._ld + 'val/', sess.graph)
                    else:
                        self._train_writer = tf.summary.FileWriter(self._ld, sess.graph)
        else:
            with tf.device('/gpu:{}'.format(self._gpu_number)):
                with tf.Session(config=self._config) as sess:
                    if seperate_validation:
                        self._train_writer = tf.summary.FileWriter(self._ld + 'train/', sess.graph)
                        self._val_writer = tf.summary.FileWriter(self._ld + 'val/')
                    else:
                        self._train_writer = tf.summary.FileWriter(self._ld, sess.graph)
        self._sess = sess
        self._network = layers

    def fit(self, X, y,
            loss_type='crossentropy',
            seperate_validation=True, validation_ratio=0.2,
            learning_rate=0.01, beta=0.0005,
            n_epochs=50, batch_size=16,
            dropout_keep_rate=1.,
            save_best_model=False):

        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param y: array-like, shape (n_samples,)
        Target vector relative to X.
        :param loss_type: string, 'rmse' or 'crossentropy'
        :param seperate_validation: bool, seperate validation set from X
        :param validation_ratio: float, between (0.0-1.0) the ratio of seperated validation (default=0.2)
        :param learning_rate: float, the learning rate
        :param beta: float, L2 regularization parameter for the weights
        :param n_epochs: int, number of epochs to train the network
        :param batch_size: int, batch size
        :param dropout_keep_rate: float, 0.6 would drop 40% of weights
        :param save_best_model: bool, whether or not to save the model with lowest loss
        :return:
        """

        X = np.array(X, np.float32)
        y = np.array(y, np.float32)
        if len(y.shape) == 1:
            y = np.expand_dims(y, 1)

        self._construct_nn(seperate_validation)

        if loss_type.lower() not in ['rmse', 'crossentropy']:
            raise Exception('invalid loss type as argument')
        else:
            loss_type = loss_type.lower()

        if seperate_validation:
            X, val_X, y, val_y = train_test_split(X, y, test_size=validation_ratio,
                                                  stratify=y, random_state=self._random_state)
            assert len(X) > 0, "The training set is empty"
            assert len(val_X) > 0, "The validation set is empty"

        train_inds = np.arange(len(X))
        num_batches_train = len(train_inds) // batch_size
        b_w = 1. / num_batches_train
        summary_op_step = int(pow(10, np.ceil(np.log10(num_batches_train))))
        batch_step = int(np.floor(summary_op_step * b_w))

        if seperate_validation:
            val_inds = np.arange(len(val_X))
            batch_size_val = len(val_inds) // num_batches_train
            val_num_batch = int(len(val_inds) / batch_size)
            summary_val_step = int(pow(10, np.ceil(np.log10(val_num_batch))))
            batch_step_val = int(np.floor(summary_val_step / val_num_batch))

        train_vars = tf.trainable_variables()

        with tf.name_scope('losses'):
            if loss_type == 'crossentropy':
                if self._is_binary:
                    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels,
                                                                                   logits=self._logits), name='ce_loss')
                else:
                    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._labels,
                                                                                   logits=self._logits), name='ce_loss')
            else:
                loss1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self._labels, self._logits))), name='rmse_loss')
            loss2 = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in train_vars]), beta, name='l2_reg_loss')
            loss = tf.add(loss1, loss2, name='total_loss')
            tf.summary.scalar('cross_entropy_loss', loss1)
            tf.summary.scalar('l2_regularization', loss2)
            tf.summary.scalar('total', loss)

        with tf.name_scope('accuracy'):
            if self._is_binary:
                correct_prediction = tf.equal(tf.round(tf.squeeze(self._output)), tf.squeeze(self._labels))
            else:
                correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._labels, 1))
            evaluation_step = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', evaluation_step)

        summary_op = tf.summary.merge_all()
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        self._sess.run(tf.global_variables_initializer())

        start = time.time()
        sleep_time = 0.2
        self.ce_losses = []
        self.best_loss = np.inf
        for epoch in tqdm(range(n_epochs)):
            # batch_accuracies = []
            q = 0
            l = 0
            # for j in tqdm(range(num_batches_train)):
            for j in range(num_batches_train):
                train_inds, batch_inds = next_batch_shuffle(train_inds, batch_size)
                batch_features = X[batch_inds]
                batch_labels = y[batch_inds]

                _, train_summary, _loss = self._sess.run([train_op, summary_op, loss1],
                                                         feed_dict={self._input_ph: batch_features,
                                                                    self._labels: batch_labels,
                                                                    self._dropout_keep_rate: dropout_keep_rate})

                if seperate_validation:
                    val_inds, batch_inds = next_batch(val_inds, batch_size_val)
                    batch_features = val_X[batch_inds]
                    batch_labels = val_y[batch_inds]

                    batch_accuracy, val_summary, _loss = self._sess.run([evaluation_step, summary_op, loss1],
                                                                        feed_dict={self._input_ph: batch_features,
                                                                                   self._labels: batch_labels})

                    self._val_writer.add_summary(val_summary, epoch * summary_op_step + j * batch_step_val)

                l += _loss
                q += 1

                self._train_writer.add_summary(train_summary, epoch * summary_op_step + j * batch_step)

            mean_loss = l / q
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                if save_best_model:
                    self.save_pb(self._ld + 'best.pb')
            if self._verbose:
                print('%d epoch mean loss: %f' % (epoch, mean_loss))
            self.ce_losses.append(mean_loss)

            train_inds = np.arange(len(X))
            if seperate_validation:
                val_inds = np.arange(len(val_X))
            # mean_batch_summary = sess.run(mean_sampled_summary,
            #                               feed_dict={mean_sample_placeholder: np.mean(batch_accuracies)})
            # during_average_writer.add_summary(mean_batch_summary, (epoch + 1) * summary_val_step)
            time.sleep(sleep_time)
        if self._verbose:
            print('The training took {} seconds'.format(time.time() - start - epoch * sleep_time))
        return

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
        if self._is_binary:
            predictions0 = 1 - predictions
            predictions = np.expand_dims(predictions, 1)
            predictions0 = np.expand_dims(predictions0, 1)
            predictions = np.concatenate((predictions0, predictions), axis=1)
        return predictions

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
        if self._is_binary:
            return accuracy_score(y, y_pred)
        else:
            return accuracy_score(np.argmax(y, axis=1), y_pred)

    def save_pb(self, path_to_pb):
        freeze_save_graph(self._sess, path.basename(path_to_pb), 'output', path.dirname(path_to_pb))

    def load_pb(self, path_to_pb):
        if self._cpu_only:
            with tf.device('/cpu:{}'.format(self._cpu_number)):
                self._input_ph, self._output = load_graph(path_to_pb, ['input:0', 'output:0'])
                self._is_binary = True if int(self._output.shape[1]) == 1 else False
                self._sess = tf.Session(config=self._config)
        else:
            with tf.device('/gpu:{}'.format(self._gpu_number)):
                self._input_ph, self._output = load_graph(path_to_pb, ['input:0', 'output:0'])
                self._is_binary = True if int(self._output.shape[1]) == 1 else False
                self._sess = tf.Session(config=self._config)

        return self
