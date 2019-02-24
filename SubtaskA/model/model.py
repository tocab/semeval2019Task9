import tensorflow as tf
# from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell as LSTMCell
import pickle
import random
import numpy as np
from tensorflow.python.keras.layers import Dense, TimeDistributed
from tensorflow.contrib.rnn import DropoutWrapper
import pandas as pd
from itertools import cycle
from matplotlib import pyplot as plt

RANDOM_SEED = 4


class nn:

    def __init__(
            self,
            max_seq_len=20,
            embedding_size=768,
            one_grams_ls=128,
            cnn_ls=128,
            all_grams_ls=128,
            classifier_ls=128,
            unit_ls=512,
            epochs=1000
    ):
        sess = tf.Session()
        self.sess = sess
        self.max_seq_len = max_seq_len

        self.one_grams_ls = one_grams_ls
        self.all_grams_ls = all_grams_ls
        self.classifier_ls = classifier_ls
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.cnn_ls = cnn_ls

        self.unit_ls = unit_ls

        self.x_0_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_1_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_2_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_3_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.y_ph = tf.placeholder(tf.int32)
        self.y_ph = tf.to_float(self.y_ph)

        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('g1'):
            g1_probs, g1_cross = self.build_train_graph(tf.ones_like(self.y_ph, tf.float32), self.keep_prob)
        with tf.variable_scope('g2'):
            g2_probs, g2_cross = self.build_train_graph(tf.ones_like(self.y_ph, tf.float32), self.keep_prob)
        with tf.variable_scope('g3'):
            g3_probs, g3_cross = self.build_train_graph(tf.ones_like(self.y_ph, tf.float32), self.keep_prob)

        stacked = tf.stack([g1_probs, g2_probs, g3_probs], axis=1)
        self.probs = tf.reduce_mean(stacked, axis=1)

        self.loss = tf.reduce_mean(g1_cross) + tf.reduce_mean(g2_cross) + tf.reduce_mean(g3_cross)
        self.adam_opt = tf.train.AdamOptimizer().minimize(self.loss)

        predicted = tf.reshape(tf.round(self.probs), [-1])
        TP = tf.count_nonzero(tf.multiply(predicted, self.y_ph))
        TN = tf.count_nonzero(tf.multiply((predicted - 1), (self.y_ph - 1)))
        FP = tf.count_nonzero(tf.multiply(predicted, (self.y_ph - 1)))
        FN = tf.count_nonzero(tf.multiply((predicted - 1), self.y_ph))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        self.f1 = 2 * precision * recall / (precision + recall)

        self.saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

    def build_train_graph(self, weights, keep_prob):
        train_outputs = self.build_graph(self.x_0_ph, self.x_1_ph, self.x_2_ph, self.x_3_ph, keep_prob)
        probs = tf.nn.sigmoid(train_outputs)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(self.y_ph, [-1, 1]),
            logits=train_outputs
        )
        cross_entropy = cross_entropy * weights
        loss = tf.reduce_mean(cross_entropy)

        return probs, cross_entropy

    def build_graph(self, x_0_ph, x_1_ph, x_2_ph, x_3_ph, keep_prob, reuse=False):
        with tf.variable_scope('layer_0_lstm', reuse=reuse):
            output_0 = self.create_big_layer_lstm(x_0_ph, keep_prob, reuse=reuse)
        with tf.variable_scope('layer_1_lstm', reuse=reuse):
            output_1 = self.create_big_layer_lstm(x_1_ph, keep_prob, reuse=reuse)
        with tf.variable_scope('layer_2_lstm', reuse=reuse):
            output_2 = self.create_big_layer_lstm(x_2_ph, keep_prob, reuse=reuse)
        with tf.variable_scope('layer_3_lstm', reuse=reuse):
            output_3 = self.create_big_layer_lstm(x_3_ph, keep_prob, reuse=reuse)
        with tf.variable_scope('layer_0_cnn', reuse=reuse):
            output_0_cnn = self.create_big_layer_cnn(tf.stack([x_0_ph, x_1_ph, x_2_ph, x_3_ph], axis=-1), keep_prob,
                                                     reuse=reuse)

        # Concat all outputs
        all_outputs = tf.concat(
            [
                output_0,
                output_1,
                output_2,
                output_3,
                output_0_cnn
            ],
            axis=1
        )

        # Make classifier
        outputs = tf.nn.dropout(
            tf.layers.dense(inputs=all_outputs, units=self.classifier_ls, activation=tf.nn.relu, reuse=reuse,
                            name='classifier_0'), keep_prob=keep_prob)
        outputs = tf.nn.dropout(
            tf.layers.dense(inputs=outputs, units=self.classifier_ls, activation=tf.nn.relu, reuse=reuse,
                            name='classifier_1'), keep_prob=keep_prob)
        outputs = tf.layers.dense(inputs=outputs, units=1, reuse=reuse, name='classifier_2')

        return outputs

    def create_big_layer_cnn(self, input_tensor, keep_prob, reuse=False):

        # n grams
        one_grams_output_4 = self.n_gram_layer_conv_4(1, input_tensor, keep_prob, reuse=reuse)
        two_grams_output_4 = self.n_gram_layer_conv_4(2, input_tensor, keep_prob, reuse=reuse)
        three_grams_output_4 = self.n_gram_layer_conv_4(3, input_tensor, keep_prob, reuse=reuse)
        four_grams_output_4 = self.n_gram_layer_conv_4(4, input_tensor, keep_prob, reuse=reuse)
        five_grams_output_4 = self.n_gram_layer_conv_4(5, input_tensor, keep_prob, reuse=reuse)

        # Concat all outputs
        all_outputs = tf.concat(
            [
                one_grams_output_4,
                two_grams_output_4,
                three_grams_output_4,
                four_grams_output_4,
                five_grams_output_4
            ],
            axis=1
        )

        # Make classifier
        outputs = tf.nn.dropout(tf.layers.dense(inputs=all_outputs, units=self.classifier_ls * 4, activation=tf.nn.relu,
                                                name='cnn_classifier_0', reuse=reuse), keep_prob=keep_prob)
        outputs = tf.nn.dropout(tf.layers.dense(inputs=outputs, units=self.classifier_ls * 4, activation=tf.nn.relu,
                                                name='cnn_classifier_1', reuse=reuse), keep_prob=keep_prob)
        outputs = tf.nn.dropout(tf.layers.dense(inputs=outputs, units=self.classifier_ls * 4, activation=tf.nn.relu,
                                                name='cnn_classifier_2', reuse=reuse), keep_prob=keep_prob)

        return outputs

    def create_big_layer_lstm(self, input_tensor, keep_prob, reuse=False):
        # One grams
        one_grams_output = TimeDistributed(Dense(self.one_grams_ls, activation='sigmoid'))(input_tensor)
        one_grams_output = tf.reshape(one_grams_output, [-1, self.max_seq_len * self.one_grams_ls])
        one_grams_output = tf.layers.dense(inputs=one_grams_output, units=self.unit_ls, activation=tf.nn.relu,
                                           reuse=reuse)

        # All grams
        cell_fw = DropoutWrapper(
            LSTMCell(self.all_grams_ls),
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob,
            state_keep_prob=keep_prob,
            variational_recurrent=True,
            input_size=self.embedding_size,
            dtype=tf.float32
        )
        cell_bw = DropoutWrapper(
            LSTMCell(self.all_grams_ls),
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob,
            state_keep_prob=keep_prob,
            variational_recurrent=True,
            input_size=self.embedding_size,
            dtype=tf.float32
        )

        with tf.variable_scope('all_grams_lstm'):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                inputs=input_tensor,
                dtype=tf.float32,
                time_major=False
            )

        outputs = tf.concat(outputs, 2)
        all_grams_outputs = tf.keras.layers.GlobalMaxPool1D()(outputs)
        all_grams_outputs = tf.nn.dropout(
            tf.layers.dense(inputs=all_grams_outputs, units=self.unit_ls, activation=tf.nn.relu), keep_prob)
        all_grams_outputs = tf.nn.dropout(
            tf.layers.dense(inputs=all_grams_outputs, units=self.unit_ls, activation=tf.nn.relu), keep_prob)

        # Concat all outputs
        all_outputs = tf.concat(
            [
                one_grams_output,
                all_grams_outputs
            ],
            axis=1
        )

        # Make classifier
        outputs = tf.nn.dropout(tf.layers.dense(inputs=all_outputs, units=self.classifier_ls, activation=tf.nn.relu),
                                keep_prob)
        outputs = tf.nn.dropout(tf.layers.dense(inputs=outputs, units=self.classifier_ls, activation=tf.nn.relu),
                                keep_prob)

        return outputs

    def n_gram_layer_conv_4(self, size, input_tensor, keep_prob, reuse=False):

        with tf.variable_scope('n_gram_layer_conv_4' + str(size), reuse=reuse):
            outputs = tf.layers.conv2d(input_tensor, self.cnn_ls, [size, self.embedding_size], reuse=reuse,
                                       name='conv_' + str(size))
            # outputs = tf.layers.batch_normalization(outputs, axis=3, reuse=reuse, name='batch_norm_' + str(size))
            outputs = tf.nn.relu(outputs)

            outputs_3d = tf.reshape(outputs, [-1, (self.max_seq_len - size + 1), self.cnn_ls])
            max_pool = tf.keras.layers.GlobalMaxPool1D()(outputs_3d)
            avg_pool = tf.keras.layers.GlobalAvgPool1D()(outputs_3d)

            outputs_max_avg = tf.nn.dropout(
                tf.layers.dense(
                    inputs=tf.concat([max_pool, avg_pool], axis=-1),
                    units=1024,
                    activation=tf.nn.relu,
                    reuse=reuse,
                    name='dense_max_avg_0_' + str(size)
                ),
                keep_prob
            )
            outputs_max_avg = tf.nn.dropout(
                tf.layers.dense(
                    inputs=outputs_max_avg,
                    units=1024,
                    activation=tf.nn.relu,
                    reuse=reuse,
                    name='dense_max_avg_1_' + str(size)
                ),
                keep_prob
            )

            outputs_2d = tf.reshape(outputs, [-1, (self.max_seq_len - size + 1) * self.cnn_ls])
            # outputs_2d = tf.nn.dropout(
            #     tf.layers.dense(
            #         inputs=outputs_2d,
            #         units=128,
            #         activation=tf.nn.relu,
            #         reuse=reuse,
            #         name='dim_reduce_2d' + str(size)
            #     ),
            #     keep_prob
            # )

            outputs = tf.concat([outputs_max_avg, outputs_2d], axis=-1)

            outputs = tf.nn.dropout(
                tf.layers.dense(
                    inputs=outputs,
                    units=self.unit_ls,
                    activation=tf.nn.relu,
                    reuse=reuse,
                    name='dense_after_cnn_0_' + str(size),
                ),
                keep_prob
            )
            outputs = tf.nn.dropout(
                tf.layers.dense(
                    inputs=outputs,
                    units=self.unit_ls,
                    activation=tf.nn.relu,
                    reuse=reuse,
                    name='dense_after_cnn_1_' + str(size)
                ),
                keep_prob
            )
            outputs = tf.nn.dropout(
                tf.layers.dense(
                    inputs=outputs,
                    units=self.unit_ls,
                    activation=tf.nn.relu,
                    reuse=reuse,
                    name='dense_after_cnn_2_' + str(size)
                ),
                keep_prob
            )

        return outputs

    def train(
            self,
            data,
            early_stopping_value=50,
            save_weights=False,
            verbose=False,
            batch_size=64,
            global_best_f1=0
    ):

        trial_data = pd.read_csv('submissions/goldstandard.csv', encoding='latin-1')
        trial_labels = trial_data['label'].values
        del trial_data
        trial_features = pickle.load(open("bert_features_4/trial_data_features_4_char_A.pkl", "rb"))
        trial_features = [trial_features[key] for key in range(len(trial_features))]
        trial_features_0, trial_features_1, trial_features_2, trial_features_3 = self.format_batch(trial_features)
        trial_feed_dict = {
            self.x_0_ph: trial_features_0,
            self.x_1_ph: trial_features_1,
            self.x_2_ph: trial_features_2,
            self.x_3_ph: trial_features_3,
            self.keep_prob: 1.0,

            self.y_ph: trial_labels
        }

        test_data = pd.read_csv('data/SubtaskA_EvaluationData_labeled.csv', header=None,
                                names=['id', 'sentence', 'label'], encoding='latin-1')
        test_labels = test_data['label'].values
        del test_data
        test_features = pickle.load(open("bert_features_4/subtask_A_test_data.pkl", "rb"))
        test_features = [test_features[key] for key in range(len(test_features))]
        test_features_0, test_features_1, test_features_2, test_features_3 = self.format_batch(test_features)
        test_feed_dict = {
            self.x_0_ph: test_features_0,
            self.x_1_ph: test_features_1,
            self.x_2_ph: test_features_2,
            self.x_3_ph: test_features_3,
            self.keep_prob: 1.0,

            self.y_ph: test_labels
        }

        class_0_data = []
        class_1_data = []
        for key in range(len(data)):
            if data[key]['label'] == 0:
                class_0_data.append(data[key])
            elif data[key]['label'] == 1:
                class_1_data.append(data[key])

        random.shuffle(class_0_data)
        random.shuffle(class_1_data)

        class_0_pool = cycle(class_0_data)
        class_1_pool = cycle(class_1_data)

        steps = 0
        best_f1 = 0
        early_stopping_index = 0
        val_data_points = []
        test_data_points = []
        while True:

            if steps % 2 == 0:
                train_batch = [next(class_0_pool) for _ in range(batch_size)]
            else:
                train_batch = [next(class_1_pool) for _ in range(batch_size)]

            features_0, features_1, features_2, features_3 = self.format_batch(train_batch)

            feed_dict = {
                self.x_0_ph: features_0,
                self.x_1_ph: features_1,
                self.x_2_ph: features_2,
                self.x_3_ph: features_3,
                self.keep_prob: 0.9,
                self.y_ph: [0 if steps % 2 == 0 else 1] * batch_size
            }

            loss_0, learning_rate, _ = self.sess.run([self.loss, self.learning_rate, self.adam_opt], feed_dict)
            trial_loss, trial_f1 = self.sess.run([self.loss, self.f1], trial_feed_dict)
            test_loss, test_f1 = self.sess.run([self.loss, self.f1], test_feed_dict)
            val_data_points.append(trial_f1)
            test_data_points.append(test_f1)

            print(
                'train_loss:', '{0:.3f}'.format(loss_0), '\t',
                'trial_loss:', '{0:.3f}'.format(trial_loss), '\t',
                'trial_f1:', '{0:.3f}'.format(trial_f1), '\t',
                'test_f1:', '{0:.3f}'.format(test_f1), '\t',
                'learning_rate:', '{0:.6f}'.format(learning_rate), '\t',
                steps
            )

            # early stopping
            if best_f1 < trial_f1 and steps > 50:
                early_stopping_index = 0
                best_f1 = trial_f1

                if save_weights and global_best_f1 < best_f1:
                    # Saving loop because here can occur an error
                    saved = False
                    while not saved:
                        try:
                            self.saver.save(self.sess, "best_weights/model.ckpt")
                            saved = True
                        except:
                            continue
                    if verbose:
                        print("weights saved")

            if early_stopping_index >= early_stopping_value:
                if verbose:
                    print("Early stopping")
                break

            steps += 1

            if steps > 50 and steps % 2 == 0:
                early_stopping_index += 1

        val_score_label, = plt.plot(val_data_points, label='validation score')
        test_score_label, = plt.plot(test_data_points, label='test score')
        plt.legend(handles=[val_score_label, test_score_label])
        plt.xlabel('train steps')
        plt.ylabel('f1-score')
        plt.show()

        return best_f1

    def predict(self, data, batch_size=500):

        self.saver.restore(self.sess, tf.train.latest_checkpoint("best_weights"))

        data_list = []

        for key in data:
            data_list.append(data[key])

        example_index = 0
        all_probs = []
        while example_index < len(data_list):
            test_batch = data_list[example_index: example_index + batch_size]
            features_0, features_1, features_2, features_3 = self.format_batch(test_batch)

            feed_dict = {
                self.x_0_ph: features_0,
                self.x_1_ph: features_1,
                self.x_2_ph: features_2,
                self.x_3_ph: features_3,
                self.keep_prob: 1.0
            }

            batch_probs = self.sess.run(self.probs, feed_dict=feed_dict)
            batch_probs = np.round(batch_probs)
            all_probs.append(batch_probs)

            example_index += batch_size

        all_probs = np.concatenate(all_probs)

        return all_probs

    def format_batch(self, batch):
        features_0 = []
        features_1 = []
        features_2 = []
        features_3 = []

        for example in batch:
            features_0.append(
                np.pad(
                    [example['features'][key]['values'][0] for key in example['features']],
                    [[0, self.max_seq_len], [0, 0]],
                    mode='constant'
                )[:self.max_seq_len]
            )

            features_1.append(
                np.pad(
                    [example['features'][key]['values'][1] for key in example['features']],
                    [[0, self.max_seq_len], [0, 0]],
                    mode='constant'
                )[:self.max_seq_len]
            )

            features_2.append(
                np.pad(
                    [example['features'][key]['values'][2] for key in example['features']],
                    [[0, self.max_seq_len], [0, 0]],
                    mode='constant'
                )[:self.max_seq_len]
            )

            features_3.append(
                np.pad(
                    [example['features'][key]['values'][3] for key in example['features']],
                    [[0, self.max_seq_len], [0, 0]],
                    mode='constant'
                )[:self.max_seq_len]
            )

        features_0 = np.array(features_0)
        features_1 = np.array(features_1)
        features_2 = np.array(features_2)
        features_3 = np.array(features_3)

        return features_0, features_1, features_2, features_3
