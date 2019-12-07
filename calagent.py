import logging
import random


from keras import backend as K
import tensorflow as tf
import numpy as np

import utilities

logger = logging.getLogger()

def expand_dims(x):
    return K.expand_dims(x, -1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], input_shape[1], input_shape[2], 1)

# Imitation Learning
class CALAgent:
    def __init__(self, seq_len, embedding_size, vocab_size, embedding_matrix, num_classes, policy_output, costs):
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.w_embeddings = embedding_matrix
        self.num_classes = num_classes
        self.costs = costs
        self.build_model()
        self.build_reward_network()
        self.initial()
        self.policy_output = policy_output
        self.saver = tf.train.Saver()

    def build_model(self):
        # number of class
        self.k = tf.placeholder(
            tf.int32, shape=(), name="k")
        self.c = tf.placeholder(
            tf.int32, shape=(), name="c")
        # input
        # 
        self.sequence_input = tf.placeholder(
            tf.int32, [None, self.seq_len], name="sequence_input")
        self.sent_content, sent_state_dim = self.process_sentence(self.sequence_input)

        self.pred_input = tf.placeholder(
            tf.float32, [None, self.seq_len, self.num_classes], name="marginal_prob")
        self.state_marginals, marginal_dim = self.process_prediction(self.pred_input)

        self.entropy_input = tf.placeholder(
            tf.float32, [None, self.seq_len], name="entropy_seq")
        self.state_entropy, entropy_dim = self.process_entropies(self.entropy_input)

        self.confidence_input = tf.placeholder(
            tf.float32, [None, 1], name="input_confidence")

        self.entropy_stat = tf.placeholder(
            tf.float32, [None, 3], name="entropy_stat")

        self.pays_input = tf.placeholder(
            tf.float32, [None, 1], name="pays_input")

        self.labeled_data_rep_input = tf.placeholder(
            tf.float32, [None, self.embedding_size], name="label_data_rep")
        labeled_data_rep = tf.layers.dense(inputs=self.labeled_data_rep_input, units=128, activation=tf.nn.relu, name="labeled_pool")

        inputs = tf.concat([labeled_data_rep, self.sent_content, self.state_marginals,
                            self.confidence_input, self.state_entropy, self.entropy_stat, self.pays_input], axis=1)
        self.h_fc1_all = tf.layers.dense(inputs, 256, activation=tf.nn.relu, name="policy_net_dense")
        dropout = tf.nn.dropout(
                    self.h_fc1_all, 0.5, name="policy_net_dropout")

        self.score = tf.layers.dense(inputs=dropout, units=1, name="policy_net_logits")

        # random pool 
        self.logits = tf.reshape(self.score,[-1,self.k*self.c])
        self.action_labels = tf.placeholder(tf.int32, [None, None])

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.action_labels)
        self.cost = tf.reduce_mean(self.loss)
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        # self.probabilities = tf.nn.softmax(self.logits)
        # self.predictions = tf.argmax(tf.nn.softmax(self.probabilities), 1)

    def build_reward_network(self):
        labeled_data_rep = tf.layers.dense(inputs=self.labeled_data_rep_input, units=128, activation=tf.nn.relu, name="r_lab_pool")
        self.cost_input = tf.placeholder(
            tf.float32, [None, 1], name="input_cost")
        self.reward_input = tf.placeholder(
            tf.float32, [None, 1], name="input_reward")
        inputs = tf.concat([labeled_data_rep, self.sent_content, self.state_marginals,
                            self.confidence_input, self.state_entropy, self.entropy_stat, self.cost_input], axis=1)
        self.r_fc1_all = tf.layers.dense(inputs, 256, activation=tf.nn.relu, name="reward_net_dense")
        dropout = tf.nn.dropout(
                    self.r_fc1_all, 0.5, name="reward_net_dropout")
        self.reward = tf.layers.dense(inputs=dropout, units = 1, activation=tf.nn.sigmoid, name="reward_net_logits")
        self.r_loss = tf.losses.mean_squared_error(self.reward_input, self.reward)
        #self.r_cost = tf.reduce_mean(self.loss)
        self.r_train_step = tf.train.AdamOptimizer(1e-6).minimize(self.r_loss)

    def initial(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def process_sentence(self, sequence_input):
        with tf.name_scope("sentence_cnn"):
            with tf.device('/cpu:0'):
                self.embedding_layer = tf.Variable(tf.random_uniform(
                [self.vocab_size, self.embedding_size], -1.0, 1.0), trainable=False, name="W_embedding")
                embedded_sequences = tf.nn.embedding_lookup(self.embedding_layer, sequence_input)
                self.embedded_sequences_expanded = tf.expand_dims(embedded_sequences, -1)
            filter_sizes = [3, 4, 5]
            num_filters = 128
            dropout_keep_prob = 0.5

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(
                        0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_sequences_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(list(filter_sizes))
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # Add dropout
            with tf.name_scope("dropout"):
                self.sent_content = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
        return self.sent_content, num_filters_total

    def process_prediction(self, predictions):
        with tf.name_scope("marginal_prob_cnn"):
            filter_sizes = [3]
            num_filters = 20
            dropout_keep_prob = 0.5
            pred_dim = 64
            predictions_embedding = tf.layers.dense(predictions, units=pred_dim, name="label_embedding")
            self.predictions_expanded = tf.expand_dims(predictions_embedding, -1)

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-avgpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, pred_dim, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(
                        0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.predictions_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    h = conv
                    # averagepooling over the outputs
                    pooled = tf.nn.avg_pool(
                        h,
                        ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(list(filter_sizes))
            ph_pool = tf.concat(pooled_outputs, 3)
            ph_pool_flat = tf.reshape(ph_pool, [-1, num_filters_total])
            # Add dropout
            with tf.name_scope("dropout"):
                self.state_marginals = tf.nn.dropout(
                    ph_pool_flat, dropout_keep_prob)
        return self.state_marginals, num_filters_total

    def process_entropies(self, entropies):
        with tf.name_scope("entropy_cnn"):
            filter_sizes = [2, 3, 4]
            num_filters = 20
            dropout_keep_prob = 0.5
            ent_dim = 64
            entropy_expand = tf.expand_dims(entropies, -1)
            entropy_embedding = tf.layers.dense(entropy_expand, units=ent_dim, name="entropy_embedding")
            self.entropy_expanded = tf.expand_dims(entropy_embedding, -1)

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, ent_dim, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(
                        0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.entropy_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # averagepooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(list(filter_sizes))
            ph_pool = tf.concat(pooled_outputs, 3)
            ph_pool_flat = tf.reshape(ph_pool, [-1, num_filters_total])
            # Add dropout
            with tf.name_scope("dropout"):
                self.state_entropy = tf.nn.dropout(
                    ph_pool_flat, dropout_keep_prob)
        return self.state_entropy, num_filters_total


    def predict_action(self, k, c, sent_contents, marginal_prob, confidence_score, label_data, entropy_input, entropy_stat, pays_input):
        scores = self.sess.run(self.score,
                                               feed_dict={self.k: k,
                                                          self.c: c, 
                                                          self.sequence_input: sent_contents,
                                                          self.pred_input: marginal_prob,
                                                          self.confidence_input: confidence_score,
                                                          self.labeled_data_rep_input: label_data,
                                                          self.entropy_input: entropy_input,
                                                          self.entropy_stat: entropy_stat,
                                                          self.pays_input: pays_input})
        # props = utilities.softmax(scores)
        # scores of dimension k * c
        return np.argmax(scores) % k, np.argmax(scores) // k

    def predict(self, k, state, payments):
        sent_content = state[0]
        marginal_prob = state[1]
        confidence_score = state[2]
        labeled_data = state[3]
        entropy_input = state[4]
        entropy_stat = state[5]
        pays_input = state[6]
        return self.predict_action(k, len(payments), sent_content, marginal_prob, confidence_score, labeled_data, entropy_input, entropy_stat, pays_input)

    def predict_reward(self, reward, sent_contents, marginal_prob, confidence_score, label_data, entropy_input, entropy_stat, cost):
        scores = self.sess.run(self.reward,
                                               feed_dict={self.sequence_input: sent_contents,
                                                          self.pred_input: marginal_prob,
                                                          self.confidence_input: confidence_score,
                                                          self.labeled_data_rep_input: label_data,
                                                          self.entropy_input: entropy_input,
                                                          self.entropy_stat: entropy_stat,
                                                          self.cost_input: cost})
        return scores

    def train_policy(self, k, c, states, actions):
        # Why no rewards
        BATCH_SIZE=16
        epochs = 5
        # sentent_contents
        sent_contents = np.array([s[0] for s in states])
        marginal_prob = np.array([s[1] for s in states])
        confidence_score = np.array([s[2] for s in states])
        label_data = np.array([s[3] for s in states])
        entropies  = np.array([s[4] for s in states])
        ent_stat  = np.array([s[5] for s in states])
        pays_input  = np.array([s[6] for s in states])
        data_len = len(states)
        max_step = int(data_len / BATCH_SIZE) + 1

        seq_length = np.shape(sent_contents)[-1]
        num_class = np.shape(marginal_prob)[-1]
        label_data_dim = np.shape(label_data)[-1]
        ent_stat_num = np.shape(ent_stat)[-1]
        for epoch in range(epochs):
            indices = np.arange(data_len)
            np.random.shuffle(indices)
            sent_contents = sent_contents[indices]
            marginal_prob = marginal_prob[indices]
            confidence_score = confidence_score[indices]
            label_data = label_data[indices]
            entropies = entropies[indices]
            ent_stat = ent_stat[indices]
            pays_input = pays_input[indices]
            actions = actions[indices]
            for step in range(max_step):
                from_idx = step * BATCH_SIZE
                to_idx = (step + 1) * BATCH_SIZE
                if to_idx >= data_len:
                    to_idx = data_len - 1

                minibatch_sent_contents = np.reshape(sent_contents[from_idx:to_idx],[-1, seq_length])
                minibatch_marginal_prob = np.reshape(marginal_prob[from_idx:to_idx],[-1, seq_length, num_class])
                minibatch_confidence_score = np.reshape(confidence_score[from_idx:to_idx],[-1,1])
                minibatch_label_data = np.reshape(label_data[from_idx:to_idx],[-1,label_data_dim])
                minibatch_entropy_input = np.reshape(entropies[from_idx:to_idx],[-1,seq_length])
                minibatch_ent_stat = np.reshape(ent_stat[from_idx:to_idx], [-1, ent_stat_num])
                minibatch_pays_input = np.reshape(pays_input[from_idx:to_idx], [-1, 1])
                minibatch_actions = actions[from_idx:to_idx]
                train_loss, train_step, logits, loss, cost, score = self.sess.run([self.cost, self.train_step, self.logits, self.loss, self.cost, self.score],
                                                       feed_dict={ self.k: k, self.c: c, self.sequence_input: minibatch_sent_contents,
                                                              self.pred_input: minibatch_marginal_prob,
                                                              self.confidence_input: minibatch_confidence_score,
                                                              self.labeled_data_rep_input: minibatch_label_data,
                                                              self.entropy_stat: minibatch_ent_stat,
                                                              self.entropy_input: minibatch_entropy_input,
                                                              self.pays_input: minibatch_pays_input,
                                                              self.action_labels: minibatch_actions})
                logger.info(" >>> [{}/{}] Training policy step = {} loss = {}".
                    format(epoch, step, train_step, train_loss))
        logger.info(" >>> Save policy to {}".format(self.policy_output))
        self.save_model()

    def train_reward(self, states, costs, final_rewards):
        BATCH_SIZE=16
        epochs = 5
        # sentent_contents
        sent_contents = np.array([s[0] for s in states])
        marginal_prob = np.array([s[1] for s in states])
        confidence_score = np.array([s[2] for s in states])
        label_data = np.array([s[3] for s in states])
        entropies  = np.array([s[4] for s in states])
        ent_stat  = np.array([s[5] for s in states])
        costs = np.array(costs)
        final_rewards = np.array(final_rewards)
        data_len = len(states)
        max_step = int(data_len / BATCH_SIZE) + 1

        seq_length = np.shape(sent_contents)[-1]
        num_class = np.shape(marginal_prob)[-1]
        label_data_dim = np.shape(label_data)[-1]
        ent_stat_num = np.shape(ent_stat)[-1]
        for epoch in range(epochs):
            indices = np.arange(data_len)
            np.random.shuffle(indices)
            sent_contents = sent_contents[indices]
            marginal_prob = marginal_prob[indices]
            confidence_score = confidence_score[indices]
            label_data = label_data[indices]
            entropies = entropies[indices]
            ent_stat = ent_stat[indices]
            costs = costs[indices]
            final_rewards = final_rewards[indices]
            for step in range(max_step):
                from_idx = step * BATCH_SIZE
                to_idx = (step + 1) * BATCH_SIZE
                if to_idx >= data_len:
                    to_idx = data_len - 1

                minibatch_sent_contents = np.reshape(sent_contents[from_idx:to_idx],[-1, seq_length])
                minibatch_marginal_prob = np.reshape(marginal_prob[from_idx:to_idx],[-1, seq_length, num_class])
                minibatch_confidence_score = np.reshape(confidence_score[from_idx:to_idx],[-1,1])
                minibatch_label_data = np.reshape(label_data[from_idx:to_idx],[-1,label_data_dim])
                minibatch_entropy_input = np.reshape(entropies[from_idx:to_idx],[-1,seq_length])
                minibatch_ent_stat = np.reshape(ent_stat[from_idx:to_idx], [-1, ent_stat_num])
                minibatch_costs = np.reshape(costs[from_idx:to_idx],[-1,1])
                minibatch_final_rewards = np.reshape(final_rewards[from_idx:to_idx],[-1,1])
                train_loss, train_step, reward = self.sess.run([self.r_loss, self.r_train_step, self.reward],
                                                       feed_dict={self.sequence_input: minibatch_sent_contents,
                                                              self.pred_input: minibatch_marginal_prob,
                                                              self.confidence_input: minibatch_confidence_score,
                                                              self.labeled_data_rep_input: minibatch_label_data,
                                                              self.entropy_stat: minibatch_ent_stat,
                                                              self.entropy_input: minibatch_entropy_input,
                                                              self.cost_input: minibatch_costs,
                                                              self.reward_input: minibatch_final_rewards})
                logger.info(" >>> [{}/{}] Training reward net step = {} loss = {}".
                    format(epoch, step, train_step, train_loss))
        logger.info(" >>> Save reward net to {}".format(self.policy_output))
        self.save_model()


    def update_embeddings(self, embedding_table):
        self.w_embeddings = embedding_table
        self.vocab_size = len(self.w_embeddings)
        self.embedding_size = len(self.w_embeddings[0])
        logger.info("Assigning new word embeddings")
        logger.info("New size {}".format(self.vocab_size))
        self.sess.run(self.embedding_layer.assign(self.w_embeddings))
        self.time_step = 0

    def save_model(self):
        self.saver.save(self.sess, self.policy_output)

    def load_model(self, checkpoint, selected_modules=None, black_list=[]):
        if selected_modules is None:
            selected_modules = ["sentence_cnn","marginal_prob_cnn","labeled_pool","r_lab_pool", "policy_net", "label_embedding",
                                "entropy_cnn", "entropy_embedding", "reward_net"]

        def contains_modules(name):
            for module in selected_modules:
                if module in name:
                    return True

        vars = [v for v in tf.trainable_variables() if contains_modules(v.name)]

        # new_saver = tf.train.import_meta_graph(checkpoint + ".meta")
        with tf.Session() as sess:
            tf.train.Saver(vars).restore(sess, checkpoint)
            logger.info('retrieved parameters ({})'.format(len(vars)))
            for var in sorted(vars, key=lambda var: var.name):
                logger.info('  {} {}'.format(var.name, var.get_shape()))
            # new_saver.restore(sess, checkpoint)
            # g = tf.get_default_graph()
            # x = g.get_all_collection_keys()

