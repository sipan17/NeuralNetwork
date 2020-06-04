import os
import warnings
from shutil import rmtree

import numpy as np
import tensorflow as tf


def validate(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def find_all_files(path, extensions=(), exclude=()):
    # Find all images with given extensions
    path = os.path.abspath(path) + '/'
    if len(extensions) > 0:
        files = [path + item for item in os.listdir(path) if item.split('.')[-1].lower() in extensions]
    else:
        files = [path + item for item in os.listdir(path) if os.path.isfile(path + item)]
    dirs = [item for item in os.listdir(path) if (os.path.isdir(path + item) and item not in exclude)]
    for item in dirs:
        files += find_all_files(path + item, extensions)
    return files


def load_graph(graph_path, return_elements=None):
    # Creates graph from saved graph_def.pb
    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        output_nodes = tf.compat.v1.import_graph_def(graph_def, return_elements=return_elements)
        return output_nodes


def freeze_save_graph(sess, name, output_node, log_dir):
    for node in sess.graph.as_graph_def().node:
        node.device = ""
    variable_graph_def = sess.graph.as_graph_def()
    optimized_net = tf.compat.v1.graph_util.convert_variables_to_constants(sess, variable_graph_def, [output_node])
    tf.io.write_graph(optimized_net, log_dir, name, False)


def next_batch(data, batch_size, shuffle=False):
    """
    :param data: list or array
    :param batch_size: int, the size of the batch
    :param shuffle: bool, shuffle data before selecting the batch
    :return: tuple, (remaining data, batch data)
    """
    if len(data) <= batch_size:
        return [], data
    else:
        if shuffle:
            np.random.shuffle(data)
        return data[batch_size:], data[:batch_size]


def save(sess, saver, model_path, model_name):
    with tf.name_scope('saver'):
        saver.save(sess, os.path.join(model_path, model_name))


def load(sess, saver, model_path):
    print("Reading checkpoint ...")
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        model = os.path.basename(ckpt.model_checkpoint_path)
        next_epoch = int(model.split('_')[-1]) + 1
        saver.restore(sess, os.path.join(model_path, model))
        return True, next_epoch
    else:
        return False, 0


def clear_start(paths):
    for path in paths:
        if os.path.isdir(path):
            rmtree(path)
        validate(path)


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        std = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        mx = tf.reduce_max(var)
        mn = tf.reduce_min(var)
    tf.compat.v1.summary.scalar(name + '/mean', mean)
    tf.compat.v1.summary.scalar(name + '/stddev', std)
    tf.compat.v1.summary.scalar(name + '/max', mx)
    tf.compat.v1.summary.scalar(name + '/min', mn)
    tf.compat.v1.summary.histogram(name, var)


def construct_label(index, num_classes):
    label = np.zeros(num_classes)
    label[index] = 1
    return label
