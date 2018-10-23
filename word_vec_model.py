
# coding: utf-8

# importing libraries
import os

import tensorflow as tf
import numpy as np
import pickle
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

"""loading the data from wikipedia corpus saved as text file"""

with open("corpus/text8", "rb") as fp:
    logging.info("loading wiki data")
    wikipedia = fp.read()

logging.info("loading data finished")

"""assigning and initializing parameters"""

words = wikipedia.split(" ")[:100]
unique_words = list(set(words[1:]))
vocab_size = len(unique_words)
features_dim = 300
batch_size = 1
epochs = 100

""" generating data to be in format to be fed in skip gram architecture:
    Eg: Text: The quick fox jumps at lazy dog
    Sample: (The, quick), (The, fox), (quick,the),(quick,fox),(quick, jumps), etc"""


def form_word2id():
    """
    creating word_to_ids and id_to_word dictionary
    :return: word_to_id, id_to_word,unique_words
    """
    logging.info("creating word_to_ids and id_to_word dictionary")

    word_to_id = {}
    id_to_word = {}

    for i, word in tqdm(enumerate(unique_words)):
        word_to_id[word] = i
        id_to_word[i] = word

    logging.info("word to ids created")

    return word_to_id, id_to_word


def create_word_pairs():
    """
    creating word pairs
    :return: word_pairs
    """
    logging.info("creating word pairs")

    data = []
    window_size = 2
    for word_index, word in tqdm(enumerate(words)):
        neighbour_words = words[max(word_index - window_size, 0): min(word_index+window_size, len(words))+1]
        for nb_w in neighbour_words:
            if nb_w != word:
                data.append([word,nb_w])
    return data


def create_input_label(word_to_id, data):
    """
    seperating train x and train label
    :param word_to_id:
    :param data:
    :return: training data
    """
    logging.info("seperating train x and train label")

    train_data_x, train_data_y = [], []
    for pairs in data:
        try:
            train_data_x.append(word_to_id[pairs[0]])
            train_data_y.append(word_to_id[pairs[1]])
        except:
            pass
    num_batches = int(len(train_data_x) / batch_size)

    return train_data_x, train_data_y, num_batches


def create_hot_vector(batch):
    """
    creating one hot vector for ids for the batch which is used in the process of training
    :param batch:
    :return: one_hot_vector
    """
    one_hot_train = []
    for ids in batch:
        temporary_var = np.zeros(vocab_size)
        temporary_var[ids] = 1
        one_hot_train.append(temporary_var)
    return one_hot_train


def get_batch(batch_no, data):
    """

    :param batch_no:
    :param data:
    :return: batch of data
    """
    return data[batch_size * batch_no: batch_size * (batch_no + 1)]


"""calling all methods to receive these variables"""

word_to_id, id_to_word = form_word2id()
data = create_word_pairs()
train_data_x, train_data_y, num_batches = create_input_label(word_to_id, data)


"""creating placeholders for centre words and context words"""

with tf.name_scope("data"):
    centre_words = tf.placeholder(tf.float32, shape=[None,vocab_size], name="centre_words")
    context_words = tf.placeholder(tf.float32, shape=[None,vocab_size], name="context_words")


"""initializing trainable weights, ie weights of hidden layer and output layer"""

with tf.name_scope('embedding_matrix'):
    weight1 = tf.Variable(tf.random_uniform([vocab_size, features_dim], -1, 1), name="embedding_matrix")
with tf.name_scope('layer2_weights'):
    weight2 = tf.Variable(tf.random_uniform([features_dim,vocab_size], -1, 1), name="w2")


"""saving a global variable into cache"""

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
increment_global_step_op = tf.assign_add(global_step, 1, name="increment_global_step_op")


"""defining model architecture and computation of prediction. It consists of one hidden layer 
    with linear activation and output layer with softmax activation"""

with tf.name_scope("model"):

    one_hot = np.zeros(vocab_size)
    output1 = tf.matmul(centre_words, weight1)
    output2 = tf.matmul(output1, weight2)
    prediction = tf.nn.softmax(output2)


"""Backpropagation of error using gradient descent using adams as optimizer"""

with tf.name_scope("backprop"):
    loss = (tf.reduce_sum(-(tf.log(prediction)*context_words)))/batch_size
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)


"""saving logs for loss to see on tensorboard later"""

writer_train = tf.summary.FileWriter("../logs/")
loss_variable = tf.Variable(0.0)
summary = tf.summary.scalar("loss", loss_variable)


"""Training the model using FeedForward and BackProp.
Saving the variables weights using tf.train.Saver so that we can use it later by restoring it.
"""

saver = tf.train.Saver({"embedding_matrix": weight1, "w2": weight2, 'global_step': global_step})

with tf.Session() as sess:

    if os.path.exists("weights_stored/weights.ckpt"):
        saver.restore(sess, "weights_stored/weights.ckpt")

    sess.run(tf.global_variables_initializer())

    try:
        epoch_start = sess.run(global_step)

    except Exception as e:
        epoch_start = 0

    """run epoch and save the sess after all epochs are completed"""

    for epoch in tqdm(range(epoch_start, epochs + 1000)):
        l_batch_train = 0

        """training based on mini batching. creating one hot vector for each batch and sending them as input.
        Applying backprob and saving the loss for each batch"
        """

        for batch_no in range(num_batches):

            x_batch = get_batch(batch_no, train_data_x)
            y_batch = get_batch(batch_no, train_data_y)

            one_hot_x = create_hot_vector(x_batch)
            one_hot_y = create_hot_vector(y_batch)

            pred = sess.run([prediction], feed_dict={centre_words: one_hot_x, context_words: one_hot_y})

            l, opt = sess.run([loss, optimizer], feed_dict={centre_words: one_hot_x, context_words: one_hot_y})
            l_batch_train = l + l_batch_train

        print "loss for {} is {}".format(epoch, l_batch_train)

        """saving loss for each batch in the graph"""

        summary1 = sess.run(summary, {loss_variable: l_batch_train})
        writer_train.add_summary(summary1, epoch)
        writer_train.flush()

#     writer  = tf.summary.FileWriter("./graphs",sess.graph)

    vectors= sess.run(weight1)
    sess.run(increment_global_step_op)

    """saving the sess after epoch is completed"""

    save_path = saver.save(sess, "/tmp/weights.ckpt")


"""restore the variables(weights)
initialize the variables again but don't run initializer on them"""

with tf.name_scope('embedding_matrix'):
    weight1 = tf.Variable(tf.random_uniform([vocab_size, features_dim], -1, 1), name="embedding_matrix")

with tf.name_scope('layer2_weights'):
    weight2 = tf.Variable(tf.random_uniform([features_dim, vocab_size], -1, 1), name="w2")

saver = tf.train.Saver({"embedding_matrix": weight1,"w2":weight2})

# Use the saver object normally after that.
with tf.Session() as sess:

    saver.restore(sess, "/tmp/weights.ckpt")

"""saving the word2vec model created as a pickle file"""

with open("word_vector_model.pkl","wb") as wd_vec:
    pickle.dump(vectors,wd_vec)

with open("word_vector_model.pkl","rb") as wd_vec:
    vectors = pickle.load(wd_vec)


