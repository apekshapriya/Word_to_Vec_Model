
# coding: utf-8

# ## importing libraries
import os

import tensorflow as tf
import numpy as np
import pickle
import logging
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

# importing data
with open("corpus/text8","rb") as fp:
    logging.info("loading wiki data")
    wikipedia = fp.read()

logging.info("loading data finished")

words = wikipedia.split(" ")[:100]
# generating data to be in format to be fed in skip gram architecture

# changing word to id and id to word

logging.info("creating word_to_ids and id_to_word dictionary")

unique_words = list(set(words[1:]))
word_to_id = {}
id_to_word = {}

for i,word in tqdm(enumerate(unique_words)):
    word_to_id[word] = i
    id_to_word[i] = word

logging.info("creating word pairs")
# creating word pairs

data = []
window_size = 2
for word_index,word in tqdm(enumerate(words)):
    neighbour_words = words[max(word_index - window_size,0): min(word_index+window_size,len(words))+1]
    for nb_w in neighbour_words:
        if nb_w!=word:
            data.append([word,nb_w])

logging.info("seperating train x and train label")

print len(data)
train_data_x, train_data_y = [],[]
for pairs in data:
    try:
        train_data_x.append(word_to_id[pairs[0]])
        train_data_y.append(word_to_id[pairs[1]])
    except:
        pass

print len(train_data_y), len(train_data_x)
# creating one hot vector for ids


def create_hot_vector(batch):
    one_hot_train = []
    for ids in batch:
        temporary_var = np.zeros(vocab_size)
        temporary_var[ids] = 1
        one_hot_train.append(temporary_var)
    return one_hot_train

# Defining tf model

# initializing parameters


vocab_size = len(unique_words)
print "vocab", vocab_size
features_dim = 300
batch_size = 1
num_batches = int(len(train_data_x)/batch_size)
epochs = 100
print num_batches


# creating placeholders


with tf.name_scope("data"):
    centre_words = tf.placeholder(tf.float32, shape=[None,vocab_size], name="centre_words")
    context_words = tf.placeholder(tf.float32, shape=[None,vocab_size], name="context_words")


#  initializing trainable weights

with tf.name_scope('embedding_matrix'):
    weight1 = tf.Variable(tf.random_uniform([vocab_size, features_dim], -1, 1), name="embedding_matrix")
with tf.name_scope('layer2_weights'):
    weight2 = tf.Variable(tf.random_uniform([features_dim,vocab_size], -1, 1), name="w2")

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
increment_global_step_op = tf.assign_add(global_step, 1, name="increment_global_step_op")


def get_batch(batch_no,data):
    """
    params: epoch
    params:onehot as data
    """

    return data[batch_size * batch_no: batch_size * (batch_no + 1)]


# defining model architecture and computation of prediction

with tf.name_scope("model"):
    one_hot = np.zeros(vocab_size)

    output1 = tf.matmul(centre_words,weight1)
    output_activation = tf.nn.relu(output1)
    output2 = tf.matmul(output_activation,weight2)
    prediction = tf.nn.softmax(output2)
    print prediction


with tf.name_scope("backprop"):
    print context_words
    print prediction
    loss = (tf.reduce_sum(-(tf.log(prediction)*context_words)))/batch_size
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

writer_train = tf.summary.FileWriter("../logs/")
loss_variable = tf.Variable(0.0)
tf.summary.scalar("loss", loss_variable)


# Training the model

# save the variables weights using tf.train.Saver so that we can use it later by restoring it


saver = tf.train.Saver({"embedding_matrix": weight1, "w2": weight2, 'global_step': global_step})

with tf.Session() as sess:
    if os.path.exists("weights_stored/weights.ckpt"):
        saver.restore(sess, "weights_stored/weights.ckpt")
    sess.run(tf.global_variables_initializer())
    try:
        epoch_start = sess.run(global_step)
    except:
        raise ValueError("failed to restore model")
        epoch_start = 0

    for epoch in tqdm(range(epoch_start, epochs + 1000)):
        l_batch_train = 0
        print num_batches
        for batch_no in range(num_batches):

            x_batch = get_batch(batch_no, train_data_x)
            y_batch = get_batch(batch_no, train_data_y)

            one_hot_x = create_hot_vector(x_batch)
            one_hot_y = create_hot_vector(y_batch)
            print len(one_hot_y)
            print len(one_hot_x)
            print batch_no
            #logging.info("one hot created")
            pred = sess.run([prediction], feed_dict={centre_words: one_hot_x, context_words: one_hot_y})

            l, opt = sess.run([loss, optimizer], feed_dict={centre_words: one_hot_x, context_words: one_hot_y})

            l_batch_train = l + l_batch_train

           # print "loss for {} is {}".format(batch_no, l)
        print "loss for {} is {}".format(epoch, l_batch_train)
        summary1 = sess.run(summary, {loss_variable: l_batch_train})
        writer_train.add_summary(summary1, epoch)
        writer_train.flush()
#     writer  = tf.summary.FileWriter("./graphs",sess.graph)

    vectors= sess.run(weight1)
    sess.run(increment_global_step_op)
    save_path = saver.save(sess, "/tmp/weights.ckpt") #saves the variables mentioned in the given pathwith tf.name_scope('embedding_matrix'):


# ### restore the variables(weights)


##initialize the variables again but dont run initializer on them

with tf.name_scope('embedding_matrix'):
    weight1 = tf.Variable(tf.random_uniform([vocab_size,features_dim],-1,1),name ="embedding_matrix")
with tf.name_scope('layer2_weights'):
    weight2 = tf.Variable(tf.random_uniform([features_dim,vocab_size],-1,1),name = "w2")
    
saver = tf.train.Saver({"embedding_matrix": weight1,"w2":weight2})

# Use the saver object normally after that.
with tf.Session() as sess:
  
    saver.restore(sess, "/tmp/weights.ckpt")
#     print("w1 : %s" % weight1.eval())
#     print("w2", weight2.eval())



with open("word_vector_model.pkl","wb") as wd_vec:
    pickle.dump(vectors,wd_vec)

with open("word_vector_model.pkl","rb") as wd_vec:
    vectors = pickle.load(wd_vec)


