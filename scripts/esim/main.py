# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ESIM"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import collections
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from transformers import BertTokenizer

from data_iterator import TextIterator


flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string("train_file", None, "The train file path.")
flags.DEFINE_string("model_name_or_path", None, "The pretrained model name or path")
flags.DEFINE_string("valid_file", None, "The validation file path.")
flags.DEFINE_string("test_file", None, "The test file path.")
flags.DEFINE_string("vocab_file", None, "The vocabulary file.")
flags.DEFINE_string("output_dir", None,
                    "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("cache_dir", None,
                    "The cache directory for bert tokenizer")

# Other parameters
flags.DEFINE_string("embedding_file", None, "The pre-trained embedding file path.")

flags.DEFINE_bool("fix_embedding", True, "Whether to fix embedding during training.")
flags.DEFINE_bool("use_cudnn", False, "Whether to cudnn version BiLSTM.")

flags.DEFINE_integer("disp_freq", 10, "The display frequence of log information.")
flags.DEFINE_integer("maxlen_1", 256, "The maximum total first input sequence length.")
flags.DEFINE_integer("maxlen_2", 256, "The maximum total second input sequence length.")
flags.DEFINE_integer("hidden_size", 300, "The hidden size of hidden states for BiLSTM and MLP.")
flags.DEFINE_integer("dim_word", 768, "The dim of word embedding.")
flags.DEFINE_integer("patience", 1, "The early stopping patience.")
flags.DEFINE_integer("vocab_size", 21128, "The vocab size.")
flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")
flags.DEFINE_integer("valid_batch_size", 16, "Total batch size for validation.")
flags.DEFINE_integer("test_batch_size", 16, "Total batch size for test.")
flags.DEFINE_integer("max_train_epochs", 5, "Max number of training epochs to perform.")
flags.DEFINE_integer("num_labels", 2, "Number of labels.")

flags.DEFINE_float("learning_rate", 2e-4, "The initial learning rate for Adam.")
flags.DEFINE_float("clip_c", 10., "Gradient clipping threshold.")


def prepare_data(instance):
    """ padding the data with minibatch
    Args:
        instance: [list, list, list] for [labels, seqs_x, seqs_y]

    Return:
        x: int64 numpy.array of shape [seq_length_x, batch_size].
        x_mask: float32 numpy.array of shape [seq_length_x, batch_size].
        y: int64 numpy.array of shape [seq_length_y, batch_size].
        y_mask: float32 numpy.array of shape [seq_length_y, batch_size].
        l: int64 numpy.array of shape [batch_size, ].
    """

    seqs_x = []
    seqs_y = []
    labels = []
    # queries = []

    for ins in instance:
        # queries.append(ins[0])
        labels.append(ins[0])
        seqs_x.append(ins[1])
        seqs_y.append(ins[2])

    # lengths_q = [len(s) for s in queries]
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    maxlen_1 = FLAGS.maxlen_1
    maxlen_2 = FLAGS.maxlen_2

    # new_seqs_q = []
    new_seqs_x = []
    new_seqs_y = []
    # new_lengths_q = []
    new_lengths_x = []
    new_lengths_y = []
    # for l_q, s_q, l_x, s_x, l_y, s_y in zip(lengths_q, queries, lengths_x, seqs_x, lengths_y, seqs_y):
    for l_x, s_x, l_y, s_y, l in zip(lengths_x, seqs_x, lengths_y, seqs_y, labels):
        if l_x > maxlen_1:
            new_seqs_x.append([s_x[0]] + s_x[1:(maxlen_1-1)] + [s_x[-1]])
            new_lengths_x.append(maxlen_1)
        else:
            new_seqs_x.append(s_x)
            new_lengths_x.append(l_x)
        if l_y > maxlen_2:
            new_seqs_y.append([s_y[0]] + s_y[1:(maxlen_2-1)] + [s_y[-1]])
            new_lengths_y.append(maxlen_2)
        else:
            new_seqs_y.append(s_y)
            new_lengths_y.append(l_y)

    # lengths_q = new_lengths_q
    # queries = new_seqs_q
    lengths_x = new_lengths_x
    seqs_x = new_seqs_x
    lengths_y = new_lengths_y
    seqs_y = new_seqs_y

    if len(lengths_x) < 1 or len(lengths_y) < 1:
        return None

    n_samples = len(seqs_x)
    # maxlen_q = numpy.max(lengths_q)
    maxlen_x = np.max(lengths_x)
    maxlen_y = np.max(lengths_y)

    # q = numpy.zeros((maxlen_q, n_samples)).astype("int64")
    x = np.zeros((maxlen_x, n_samples)).astype("int64")
    y = np.zeros((maxlen_y, n_samples)).astype("int64")
    # q_mask = numpy.zeros((maxlen_q, n_samples)).astype("float32")
    x_mask = np.zeros((maxlen_x, n_samples)).astype("float32")
    y_mask = np.zeros((maxlen_y, n_samples)).astype("float32")

    # for idx, (s_q, s_x, s_y) in enumerate(zip(queries, seqs_x, seqs_y)):
    for idx, (s_x, s_y) in enumerate(zip(seqs_x, seqs_y)):
        # q[:lengths_x[idx], idx] = s_q
        # q_mask[:lengths_x[idx], idx] = 1.
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx], idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx], idx] = 1.

    return (x, x_mask, y, y_mask, labels)


def bilstm_layer_cudnn(input_data, num_layers, rnn_size, keep_prob=1.):
    """Multi-layer BiLSTM cudnn version, faster
    Args:
        input_data: float32 Tensor of shape [seq_length, batch_size, dim].
        num_layers: int64 scalar, number of layers.
        rnn_size: int64 scalar, hidden size for undirectional LSTM.
        keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers 

    Return:
        output: float32 Tensor of shape [seq_length, batch_size, dim * 2]

    """
    with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=rnn_size,
            input_mode="linear_input",
            direction="bidirectional",
            dropout=1 - keep_prob)

        # to do, how to include input_mask
        outputs, output_states = lstm(inputs=input_data)

    return outputs


def bilstm_layer(input_data, num_layers, rnn_size, keep_prob=1.):
    """Multi-layer BiLSTM
    Args:
        input_data: float32 Tensor of shape [seq_length, batch_size, dim].
        num_layers: int64 scalar, number of layers.
        rnn_size: int64 scalar, hidden size for undirectional LSTM.
        keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers 

    Return:
        output: float32 Tensor of shape [seq_length, batch_size, dim * 2]

    """
    input_data = tf.transpose(input_data, [1, 0, 2])

    output = input_data
    for layer in range(num_layers):
        with tf.variable_scope('bilstm_{}'.format(layer), reuse=tf.AUTO_REUSE):

            cell_fw = tf.contrib.rnn.LSTMCell(
                rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(
                rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                              cell_bw,
                                                              output,
                                                              dtype=tf.float32)

            # Concat the forward and backward outputs
            output = tf.concat(outputs, 2)

    output = tf.transpose(output, [1, 0, 2])

    return output


def load_word_embedding():
    """ Load pre-trained word embedding
    Args:
        token_to_idx: dictionary of token to idx

    Return:
        embedding: float32 Tensor of shape [vocab_size, dim_word]

    """

    embedding_np = 0.02 * \
        np.random.randn(FLAGS.vocab_size, FLAGS.dim_word).astype("float32")

    if FLAGS.embedding_file:
        idx = 0
        with open(FLAGS.embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split(" ")
                if len(tokens) > FLAGS.dim_word:
                    tokens = tokens[1:]
                vector = np.array(tokens, dtype=np.float)
                embedding_np[idx] = vector
                idx += 1

    embedding = tf.get_variable("embedding",
                                shape=[FLAGS.vocab_size, FLAGS.dim_word],
                                initializer=tf.constant_initializer(embedding_np),
                                trainable=not FLAGS.fix_embedding)
    return embedding


def local_inference(x1, x1_mask, x2, x2_mask):
    """Local inference collected over sequences
    Args:
        x1: float32 Tensor of shape [seq_length1, batch_size, dim].
        x1_mask: float32 Tensor of shape [seq_length1, batch_size].
        x2: float32 Tensor of shape [seq_length2, batch_size, dim].
        x2_mask: float32 Tensor of shape [seq_length2, batch_size].

    Return:
        x1_dual: float32 Tensor of shape [seq_length1, batch_size, dim]
        x2_dual: float32 Tensor of shape [seq_length2, batch_size, dim]

    """

    # x1: [batch_size, seq_length1, dim].
    # x1_mask: [batch_size, seq_length1].
    # x2: [batch_size, seq_length2, dim].
    # x2_mask: [batch_size, seq_length2].
    x1 = tf.transpose(x1, [1, 0, 2])
    x1_mask = tf.transpose(x1_mask, [1, 0])
    x2 = tf.transpose(x2, [1, 0, 2])
    x2_mask = tf.transpose(x2_mask, [1, 0])

    # attention_weight: [batch_size, seq_length1, seq_length2]
    attention_weight = tf.matmul(x1, tf.transpose(x2, [0, 2, 1]))

    # calculate normalized attention weight x1 and x2
    # attention_weight_2: [batch_size, seq_length1, seq_length2]
    attention_weight_2 = tf.exp(
        attention_weight - tf.reduce_max(attention_weight, axis=2, keepdims=True))
    attention_weight_2 = attention_weight_2 * tf.expand_dims(x2_mask, 1)
    # alpha: [batch_size, seq_length1, seq_length2]
    alpha = attention_weight_2 / (tf.reduce_sum(attention_weight_2, -1, keepdims=True) + 1e-8)
    # x1_dual: [batch_size, seq_length1, dim]
    x1_dual = tf.reduce_sum(tf.expand_dims(x2, 1) * tf.expand_dims(alpha, -1), 2)
    # x1_dual: [seq_length1, batch_size, dim]
    x1_dual = tf.transpose(x1_dual, [1, 0, 2])

    # attention_weight_1: [batch_size, seq_length2, seq_length1]
    attention_weight_1 = attention_weight - tf.reduce_max(attention_weight, axis=1, keepdims=True)
    attention_weight_1 = tf.exp(tf.transpose(attention_weight_1, [0, 2, 1]))
    attention_weight_1 = attention_weight_1 * tf.expand_dims(x1_mask, 1)

    # beta: [batch_size, seq_length2, seq_length1]
    beta = attention_weight_1 / \
        (tf.reduce_sum(attention_weight_1, -1, keepdims=True) + 1e-8)
    # x2_dual: [batch_size, seq_length2, dim]
    x2_dual = tf.reduce_sum(tf.expand_dims(x1, 1) * tf.expand_dims(beta, -1), 2)
    # x2_dual: [seq_length2, batch_size, dim]
    x2_dual = tf.transpose(x2_dual, [1, 0, 2])

    return x1_dual, x2_dual


def create_model(embedding):
    """ Create the computational graph
    Args:
        embedding: float32 Tensor of shape [vocab_size, dim_word]

    Return:
        probability: float32 Tensor of shape [batch_size,]
        cost: float32 Tensor of shape [batch_size,]
    """

    # x1: int64 Tensor of shape [seq_length, batch_size].
    # x1_mask: float32 Tensor of shape [seq_length, batch_size].
    # x2: int64 Tensor of shape [seq_length, batch_size].
    # x2_mask: float32 Tensor of shape [seq_length, batch_size].
    # y: int64 Tensor of shape [batch_size,].
    # keep_rate: float32 Scalar
    # q = tf.placeholder(tf.int64, shape=[None, None], name="q")
    # q_mask = tf.placeholder(tf.float32, shape=[None, None], name="q_mask")
    x1 = tf.placeholder(tf.int64, shape=[None, None], name="x1")
    x1_mask = tf.placeholder(tf.float32, shape=[None, None], name="x1_mask")
    x2 = tf.placeholder(tf.int64, shape=[None, None], name="x2")
    x2_mask = tf.placeholder(tf.float32, shape=[None, None], name="x2_mask")
    y = tf.placeholder(tf.int64, shape=[None], name="y")
    keep_rate = tf.placeholder(tf.float32, [], name="keep_rate")

    # embedding: [length, batch, dim]
    # embq = tf.nn.embedding_lookup(embedding, q)
    emb1 = tf.nn.embedding_lookup(embedding, x1)
    emb2 = tf.nn.embedding_lookup(embedding, x2)

    # TODO: why dropout?
    # embq = tf.nn.dropout(embq, keep_rate)
    emb1 = tf.nn.dropout(emb1, keep_rate)
    emb2 = tf.nn.dropout(emb2, keep_rate)

    # embq = embq * tf.expand_dims(q_mask, -1)
    emb1 = emb1 * tf.expand_dims(x1_mask, -1)
    emb2 = emb2 * tf.expand_dims(x2_mask, -1)

    # encode the sentence pair
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        if FLAGS.use_cudnn:
            # q_enc = bilstm_layer_cudnn(embq, 1, FLAGS.hidden_size)
            x1_enc = bilstm_layer_cudnn(emb1, 1, FLAGS.hidden_size)
            x2_enc = bilstm_layer_cudnn(emb2, 1, FLAGS.hidden_size)
        else:
            # q_enc = bilstm_layer(embq, 1, FLAGS.hidden_size)
            x1_enc = bilstm_layer(emb1, 1, FLAGS.hidden_size)
            x2_enc = bilstm_layer(emb2, 1, FLAGS.hidden_size)

    # q_enc = q_enc * tf.expand_dims(q_mask, -1)
    x1_enc = x1_enc * tf.expand_dims(x1_mask, -1)
    x2_enc = x2_enc * tf.expand_dims(x2_mask, -1)

    # local inference modeling based on attention mechanism
    x1_dual, x2_dual = local_inference(x1_enc, x1_mask, x2_enc, x2_mask)
    # q1_dual, x1_dual = local_inference(q_enc, q_mask, x1_enc, x1_mask)
    # q2_dual, x2_dual = local_inference(q_enc, q_mask, x2_enc, x2_mask)

    logit = local_composition(x1_dual, x1_enc, x1_mask, x2_dual, x2_enc, x2_mask, keep_rate)
    # logit1 = local_composition(q1_dual, q_enc, q_mask, x1_dual, x1_enc, x1_mask, keep_rate)
    # logit2 = local_composition(q2_dual, q_enc, q_mask, x2_dual, x2_enc, x2_mask, keep_rate)


    # final classifier
    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)
        # s1 = tf.range(0, limit=tf.shape(logit)[0], dtype=tf.int32)
        # s2 = tf.ones_like(s1, dtype=tf.int32)
        # indices = tf.stack((s1, s2), axis=1)
        # logit_hinge = tf.gather_nd(logit, indices=indices)
        # cost = tf.losses.hinge_loss(y, logit_hinge)
        probability = tf.nn.softmax(logit)

    return probability, cost


def local_composition(x1_dual, x1_enc, x1_mask, x2_dual, x2_enc, x2_mask, keep_rate):
    x1_match = tf.concat([x1_enc, x1_dual, x1_enc * x1_dual, x1_enc - x1_dual], 2)
    x2_match = tf.concat([x2_enc, x2_dual, x2_enc * x2_dual, x2_enc - x2_dual], 2)

    # mapping high dimension feature to low dimension
    with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
        x1_match_mapping = tf.layers.dense(x1_match, FLAGS.hidden_size,
                                           activation=tf.nn.relu,
                                           name="fnn",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        x2_match_mapping = tf.layers.dense(x2_match, FLAGS.hidden_size,
                                           activation=tf.nn.relu,
                                           name="fnn",
                                           kernel_initializer=tf.truncated_normal_initializer(
                                               stddev=0.02),
                                           reuse=True)

    x1_match_mapping = tf.nn.dropout(x1_match_mapping, keep_rate)
    x2_match_mapping = tf.nn.dropout(x2_match_mapping, keep_rate)

    # inference composition
    with tf.variable_scope("composition", reuse=tf.AUTO_REUSE):
        if FLAGS.use_cudnn:
            x1_cmp = bilstm_layer_cudnn(x1_match_mapping, 1, FLAGS.hidden_size)
            x2_cmp = bilstm_layer_cudnn(x2_match_mapping, 1, FLAGS.hidden_size)
        else:
            x1_cmp = bilstm_layer(x1_match_mapping, 1, FLAGS.hidden_size)
            x2_cmp = bilstm_layer(x2_match_mapping, 1, FLAGS.hidden_size)

    logit_x1_sum = tf.reduce_sum(x1_cmp * tf.expand_dims(x1_mask, -1), 0) / \
                   tf.expand_dims(tf.reduce_sum(x1_mask, 0), 1)
    logit_x1_max = tf.reduce_max(x1_cmp * tf.expand_dims(x1_mask, -1), 0)
    logit_x2_sum = tf.reduce_sum(x2_cmp * tf.expand_dims(x2_mask, -1), 0) / \
                   tf.expand_dims(tf.reduce_sum(x2_mask, 0), 1)
    logit_x2_max = tf.reduce_max(x2_cmp * tf.expand_dims(x2_mask, -1), 0)

    logit = tf.concat([logit_x1_sum, logit_x1_max, logit_x2_sum, logit_x2_max], 1)

    # final classifier
    with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
        logit = tf.nn.dropout(logit, keep_rate)
        logit = tf.layers.dense(logit, FLAGS.hidden_size,
                                activation=tf.nn.tanh,
                                name="fnn1",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        logit = tf.nn.dropout(logit, keep_rate)
        assert FLAGS.num_labels == 2
        logit = tf.layers.dense(logit, FLAGS.num_labels,
                                activation=None,
                                name="fnn2",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    return logit


def predict_accuracy(sess, cost_op, probability_op, iterator):
    """ Caculate accuracy and loss for dataset
    Args:
        sess: tf.Session
        cost_op: cost operation
        probability_op: probability operation
        iterator: iterator of dataset

    Return:
        accuracy: float32 scalar
        loss: float32 scalar
    """
    n_done = 0
    total_correct = 0
    total_cost = 0
    for instance in iterator:
        n_done += len(instance)
        (batch_x1, batch_x1_mask, batch_x2, batch_x2_mask, batch_y) = prepare_data(instance)

        cost, probability = sess.run([cost_op, probability_op],
                                     feed_dict={"x1:0": batch_x1, "x1_mask:0": batch_x1_mask,
                                                "x2:0": batch_x2, "x2_mask:0": batch_x2_mask,
                                                "y:0": batch_y, "keep_rate:0": 1.0})

        total_correct += (np.argmax(probability, axis=1) == batch_y).sum()
        total_cost += cost.sum()

    accuracy = 1.0 * total_correct / n_done
    loss = 1.0 * total_cost / n_done

    return accuracy, loss


def average_precision(sort_data):
    """ calculate average precision (AP)
    If our returned result is 1, 0, 0, 1, 1, 1
    The precision is 1/1, 0, 0, 2/4, 3/5, 4/6
    AP = (1 + 2/4 + 3/5 + 4/6)/4 = 0.69

    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}

    Return:
        average precision
    """
    count_gold = 0
    sum_precision = 0

    for i, data in enumerate(sort_data):
        if data[1] == 1:
            count_gold += 1
            sum_precision += 1. * count_gold / (i + 1)

    ap = 1. * sum_precision / count_gold

    return ap


def reciprocal_rank(sort_data):
    """ calculate reciprocal rank
    If our returned result is 0, 0, 0, 1, 1, 1
    The rank is 4
    The reciprocal rank is 1/4
    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}

    Return:
        reciprocal rank

    """

    sort_label = [x[1] for x in sort_data]
    assert 1 in sort_label
    reciprocal_rank = 1. / (1 + sort_label.index(1))

    return reciprocal_rank


def precision_at_position_1(sort_data):
    """ calculate precision at position 1
    Precision= (Relevant_Items_Recommended in top-k) / (k_Items_Recommended)

    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}

    Return:
        precision_at_position_1

    """

    if sort_data[0][1] == 1:
        return 1
    else:
        return 0


def recall_at_position_k(sort_data, k):
    """ calculate precision at position 1
    Recall= (Relevant_Items_Recommended in top-k) / (Relevant_Items)

    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}

    Return:
        recall_at_position_k

    """

    sort_label = [s_d[1] for s_d in sort_data]
    gold_label_count = sort_label.count(1)

    select_label = sort_label[:k]
    recall_at_position_k = 1. * select_label.count(1) / gold_label_count

    return recall_at_position_k


def evaluation_one_session(data):
    """ evaluate for one session

    """

    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    ap = average_precision(sort_data)
    rr = reciprocal_rank(sort_data)
    precision1 = precision_at_position_1(sort_data)
    recall1 = recall_at_position_k(sort_data, 1)
    recall2 = recall_at_position_k(sort_data, 2)
    recall5 = recall_at_position_k(sort_data, 5)

    return ap, rr, precision1, recall1, recall2, recall5


def predict_metrics(sess, cost_op, probability_op, iterator):
    """ Caculate MAP, MRR, Precision@1, Recall@1, Recall@2, Recall@5 
    Args:
        sess: tf.Session
        cost_op: cost operation
        probability_op: probability operation
        iterator: iterator of dataset

    Return:
        metrics: float32 list, [MAP, MRR, Precision@1, Recall@1, Recall@2, Recall@5]
        scores: float32 list, probability for positive label for all instances
    """

    n_done = 0
    scores = []
    labels = []
    for instance in iterator:
        n_done += len(instance)
        (batch_x1, batch_x1_mask, batch_x2, batch_x2_mask, batch_y) = prepare_data(instance)
        cost, probability = sess.run([cost_op, probability_op],
                                     feed_dict={"x1:0": batch_x1, "x1_mask:0": batch_x1_mask,
                                                "x2:0": batch_x2, "x2_mask:0": batch_x2_mask,
                                                "y:0": batch_y, "keep_rate:0": 1.0})

        labels.extend(batch_y.tolist())
        # probability for positive label
        scores.extend(probability[:, 1].tolist())

    assert len(labels) == n_done
    assert len(scores) == n_done

    tf.logging.info("seen samples %s", n_done)

    sum_map = 0
    sum_mrr = 0
    sum_p1 = 0
    sum_r1 = 0
    sum_r2 = 0
    sum_r5 = 0
    total_num = 0
    p = 0
    t = 0
    tp = 0

    for i, (s, l) in enumerate(zip(scores, labels)):
        if i % 10 == 0:
            data = []
        data.append((float(s), int(l)))

        if i % 10 == 9:
            total_num += 1
            ap, rr, precision1, recall1, recall2, recall5 = evaluation_one_session(
                data)
            sum_map += ap
            sum_mrr += rr
            sum_p1 += precision1
            sum_r1 += recall1
            sum_r2 += recall2
            sum_r5 += recall5

    metrics = [1. * sum_map / total_num, 1. * sum_mrr / total_num, 1. * sum_p1 / total_num,
               1. * sum_r1 / total_num, 1. * sum_r2 / total_num, 1. * sum_r5 / total_num]

    return metrics, scores


def f1_metrics(sess, cost_op, probability_op, iterator):
    """ Caculate MAP, MRR, Precision@1, Recall@1, Recall@2, Recall@5
    Args:
        sess: tf.Session
        cost_op: cost operation
        probability_op: probability operation
        iterator: iterator of dataset

    Return:
        metrics: float32 list, [MAP, MRR, Precision@1, Recall@1, Recall@2, Recall@5]
        scores: float32 list, probability for positive label for all instances
    """
    n_done = 0
    scores = []
    labels = []
    while True:
        try:
            instance = iterator.next()
        except StopIteration:
            break
        if len(instance) <= 0:
            continue
        n_done += len(instance)
        (batch_x1, batch_x1_mask, batch_x2, batch_x2_mask, batch_y) = prepare_data(instance)
        cost, probability = sess.run([cost_op, probability_op],
                                     feed_dict={"x1:0": batch_x1, "x1_mask:0": batch_x1_mask,
                                                "x2:0": batch_x2, "x2_mask:0": batch_x2_mask,
                                                "y:0": batch_y, "keep_rate:0": 1.0})
        labels.extend(batch_y)
        # probability for positive label
        scores.extend(probability[:, 1].tolist())

    assert len(labels) == n_done
    assert len(scores) == n_done
    tf.logging.info("seen samples %s", n_done)

    metrics = dict()
    for threshold in np.arange(0.1, 1.0, 0.1):
        p, t, fp, fn, tp = 0, 0, 0, 0, 0
        for i, (s, l) in enumerate(zip(scores, labels)):
            if l <= 0:
                if s > threshold:
                    fp += 1
                else:
                    fn += 1
            else:
                t += 1
                if s > threshold:
                    tp += 1
            if s > threshold:
                p += 1

        denominator = tp + 0.5 * (fp + fn)
        metrics[threshold] = [tp / p if p > 0 else 0.0,
                              tp / t if t > 0 else 0.0,
                              tp / denominator if denominator > 0 else 0.0]

    return metrics, scores


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as f:
        while True:
            token = f.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def main(_):
    """Main procedure for training and test

    """

    ud_start_whole = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)

    # Load vocabulary
    tf.logging.info("***** Loading Vocabulary *****")
    # token_to_idx = load_vocab(FLAGS.vocab_file)
    tokenizer = BertTokenizer.from_pretrained(FLAGS.model_name_or_path, cache_dir=FLAGS.cache_dir,
                                              tokenize_chinese_chars=False)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load text iterator
    tf.logging.info("***** Loading Text Iterator *****")
    train = TextIterator(FLAGS.train_file, tokenizer,
                         batch_size=FLAGS.train_batch_size,
                         vocab_size=FLAGS.vocab_size,
                         shuffle=True)
    valid = TextIterator(FLAGS.valid_file, tokenizer,
                         batch_size=FLAGS.valid_batch_size,
                         vocab_size=FLAGS.vocab_size,
                         shuffle=False)
    # test = TextIterator(FLAGS.test_file, token_to_idx,
    #                     batch_size=FLAGS.test_batch_size,
    #                     vocab_size=FLAGS.vocab_size,
    #                     shuffle=False)
    # # Text iterator of training set for evaluation
    # train_eval = TextIterator(FLAGS.train_file, token_to_idx,
    #                           vocab_size=FLAGS.vocab_size, batch_size=FLAGS.train_batch_size, shuffle=False)

    # Initialize the word embedding
    tf.logging.info("***** Initialize Word Embedding *****")
    embedding = load_word_embedding()

    # Build graph
    tf.logging.info("***** Build Computation Graph *****")
    probability_op, loss_op = create_model(embedding)
    # loss_op = tf.reduce_mean(cost_op)

    lr = tf.Variable(0.0, name="learning_rate", trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    tf.logging.info("***** Trainable Variables *****")

    tvars = tf.trainable_variables()
    for var in tvars:
        tf.logging.info(" name = %s, shape = %s", var.name, var.shape)

    if FLAGS.clip_c > 0.:
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss_op, tvars), FLAGS.clip_c)

    train_op = optimizer.apply_gradients(zip(grads, tvars))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)

    # training process
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)

        uidx = 0
        bad_counter = 0
        history_errs = []

        current_lr = FLAGS.learning_rate
        sess.run(tf.assign(lr, current_lr))

        for eidx in range(FLAGS.max_train_epochs):
            tf.logging.info("***** Training at Epoch %s *****", eidx)
            n_samples = 0
            while True:
                try:
                    instance = train.next()
                except StopIteration:
                    break
                if len(instance) <= 0:
                    continue

                n_samples += len(instance)
                uidx += 1

                (batch_x1, batch_x1_mask, batch_x2, batch_x2_mask, labels) = prepare_data(instance)

                if batch_x1 is None:
                    tf.logging.info("Minibatch with zero sample")
                    uidx -= 1
                    continue

                ud_start = time.time()
                _, loss = sess.run([train_op, loss_op],
                                   feed_dict={
                    "x1:0": batch_x1, "x1_mask:0": batch_x1_mask,
                    "x2:0": batch_x2, "x2_mask:0": batch_x2_mask,
                    "y:0": labels, "keep_rate:0": 0.5})
                ud = time.time() - ud_start

                if np.mod(uidx, FLAGS.disp_freq) == 0:
                    tf.logging.info(
                        "epoch %s update %s loss %s samples/sec %s", eidx, uidx, loss, 1. * batch_x1.shape[1] / ud)

            tf.logging.info("***** Evaluation at Epoch %s *****", eidx)
            tf.logging.info("seen samples %s each epoch", n_samples)
            tf.logging.info("current learning rate: %s", current_lr)

            # validate model on validation set and early stop if necessary
            # valid_metrics, valid_scores = predict_metrics(
            #     sess, loss_op, probability_op, valid)
            valid_metrics, valid_scores = f1_metrics(
                sess, loss_op, probability_op, valid)

            # # select best model based on recall@1 of validation set
            # valid_err = 1.0 - valid_metrics[3]
            # history_errs.append(valid_err)

            # tf.logging.info(
            #     "valid set: MAP %s MRR %s Precision@1 %s Recall@1 %s Recall@2 %s Recall@5 %s", *valid_metrics)
            for threshold, (precision, recall, f1) in valid_metrics.items():
                tf.logging.info(
                    f"threshold: {threshold} Precision: {precision} Recall: {recall} f1: {f1}")

            # test_metrics, test_scores = predict_metrics(
            #     sess, cost_op, probability_op, test)
            #
            # tf.logging.info(
            #     "test set: MAP %s MRR %s Precision@1 %s Recall@1 %s Recall@2 %s Recall@5 %s", *test_metrics)
            #
            # if eidx == 0 or valid_err <= np.array(history_errs).min():
            #     best_epoch_num = eidx
            #     tf.logging.info(
            #         "saving current best model at epoch %s based on metrics on valid set", best_epoch_num)
            #     saver.save(sess, os.path.join(
            #         FLAGS.output_dir, "model_epoch_{}.ckpt".format(best_epoch_num)))

            saver.save(sess, os.path.join(
                FLAGS.output_dir, "model_epoch_{}.ckpt".format(eidx)))
            #
            # if valid_err > np.array(history_errs).min():
            #     bad_counter += 1
            #     tf.logging.info("bad_counter: %s", bad_counter)
            #
            #     current_lr = current_lr * 0.5
            #     sess.run(tf.assign(lr, current_lr))
            #     tf.logging.info(
            #         "half the current learning rate to %s", current_lr)

            # if bad_counter > FLAGS.patience:
            #     tf.logging.info("***** Early Stop *****")
            #     estop = True
            #     break

        # evaluation process
        tf.logging.info("***** Final Result ***** ")
        # tf.logging.info(
        #     "restore best model at epoch %s ", best_epoch_num)
        # saver.restore(sess, os.path.join(
        #     FLAGS.output_dir, "model_epoch_{}.ckpt".format(best_epoch_num)))
        #
        # valid_metrics, valid_scores = predict_metrics(
        #     sess, cost_op, probability_op, valid)
        # tf.logging.info(
        #     "valid set: MAP %s MRR %s Precision@1 %s Recall@1 %s Recall@2 %s Recall@5 %s", *valid_metrics)
        #
        # test_metrics, test_scores = predict_metrics(
        #     sess, cost_op, probability_op, test)
        # tf.logging.info(
        #     "test set: MAP %s MRR %s Precision@1 %s Recall@1 %s Recall@2 %s Recall@5 %s", *test_metrics)
        #
        # train_acc, train_cost = predict_accuracy(
        #     sess, cost_op, probability_op, train_eval)
        # tf.logging.info("train set: ACC %s Cost %s", train_acc, train_cost)

        ud_whole = (time.time() - ud_start_whole) / 3600

        tf.logging.info("training epochs: %s", eidx + 1)
        tf.logging.info("training duration: %s hours", ud_whole)


if __name__ == "__main__":
    flags.mark_flag_as_required("train_file")
    flags.mark_flag_as_required("model_name_or_path")
    flags.mark_flag_as_required("valid_file")
    # flags.mark_flag_as_required("test_file")
    flags.mark_flag_as_required("embedding_file")
    flags.mark_flag_as_required("output_dir")

    tf.app.run()
