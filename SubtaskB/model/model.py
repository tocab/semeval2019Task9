import tensorflow as tf
import pickle
import random
import numpy as np
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
            unit_ls=256,
            lr=0.001,
            epochs=1000,
            keep_prob=0.9,
            max_seq_len_char=150,
            vocab_size=0,
            char_embedding_size=64
    ):
        sess = tf.Session()
        self.sess = sess
        self.max_seq_len = max_seq_len
        self.max_seq_len_char = max_seq_len_char
        self.vocab_size = vocab_size
        self.char_embedding_size = char_embedding_size

        self.one_grams_ls = one_grams_ls
        self.all_grams_ls = all_grams_ls
        self.classifier_ls = classifier_ls
        self.embedding_size = embedding_size
        self.lr = lr
        self.epochs = epochs
        self.keep_prob = keep_prob
        self.cnn_ls = cnn_ls

        self.unit_ls = unit_ls

        self.x_0_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_1_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_2_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_3_ph = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])

        self.x_0_ph_unsup = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_1_ph_unsup = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_2_ph_unsup = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])
        self.x_3_ph_unsup = tf.placeholder(tf.float32, [None, self.max_seq_len, self.embedding_size])

        self.y_ph = tf.placeholder(tf.float32)
        # self.y_ph = tf.to_float(self.y_ph)

        sup_ph = [self.x_0_ph, self.x_1_ph, self.x_2_ph, self.x_3_ph]
        unsup_ph = [self.x_0_ph_unsup, self.x_1_ph_unsup, self.x_2_ph_unsup, self.x_3_ph_unsup]

        sup_features = []
        unsup_features = []
        with tf.variable_scope('feature_extractor'):
            for i, input_tensor in enumerate(zip(sup_ph, unsup_ph)):
                input_tensor_sup = tf.expand_dims(input_tensor[0], -1)
                input_tensor_unsup = tf.expand_dims(input_tensor[1], -1)

                with tf.variable_scope('conv_' + str(i)):
                    feat_sup = tf.layers.conv2d(input_tensor_sup, self.embedding_size, [1, self.embedding_size])
                    feat_sup = tf.layers.batch_normalization(feat_sup, axis=3, name='batch_norm_')
                    feat_sup = tf.nn.relu(feat_sup)
                    feat_unsup = tf.layers.conv2d(input_tensor_unsup, self.embedding_size, [1, self.embedding_size],
                                                  reuse=True)
                    feat_unsup = tf.layers.batch_normalization(feat_unsup, axis=3, name='batch_norm_', reuse=True)
                    feat_unsup = tf.nn.relu(feat_unsup)

                feat_sup = tf.reshape(feat_sup, [-1, self.max_seq_len, self.embedding_size])
                feat_unsup = tf.reshape(feat_unsup, [-1, self.max_seq_len, self.embedding_size])
                sup_features.append(feat_sup)
                unsup_features.append(feat_unsup)

        with tf.variable_scope('label_classifier'):
            repr = self.build_graph(*sup_features)
            outputs = tf.layers.dense(inputs=repr, units=1)
            label_probs = tf.nn.sigmoid(outputs)
            self.label_probs = label_probs
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.y_ph,
                logits=outputs
            )
            self.label_loss = tf.reduce_mean(cross_entropy)
            TP, FP, FN, precision, recall, self.f1 = self.calc_f1(self.label_probs, self.y_ph)

        with tf.variable_scope('domain_classifier'):
            sup_output = self.build_graph(*sup_features)
            unsup_output = self.build_graph(*unsup_features, reuse=True)

            outputs_sup = tf.layers.dense(inputs=sup_output, units=1, name='dense_1')
            outputs_unsup = tf.layers.dense(inputs=unsup_output, units=1, name='dense_1', reuse=True)

            sim_concat = tf.nn.sigmoid(tf.reshape(tf.concat([outputs_sup, outputs_unsup], 0), [-1]))
            self.unsupervised_preds = sim_concat
            sim_y = tf.reshape(tf.concat([tf.ones_like(outputs_unsup), tf.zeros_like(outputs_sup)], 0), [-1])

            TP, FP, FN, precision, recall, f1_sim = self.calc_f1(sim_concat, sim_y)
            self.f1_sim = f1_sim



        cross_entropy_sup = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(outputs_sup),
            logits=outputs_sup
        )

        cross_entropy_unsup = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(outputs_unsup),
            logits=outputs_unsup
        )

        cross_entropy_unsup_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(outputs_unsup),
            logits=outputs_unsup
        )

        self.domain_loss_real = tf.reduce_mean(cross_entropy_sup) + tf.reduce_mean(cross_entropy_unsup)
        self.domain_loss_fake = tf.reduce_mean(cross_entropy_unsup_fake)

        tvars = tf.trainable_variables()
        feature_extractor_vars = [var for var in tvars if 'feature_extractor' in var.name]
        label_classifier_vars = [var for var in tvars if 'label_classifier' in var.name]
        label_classifier_controller_vars = [var for var in tvars if 'label_classifier_controller' in var.name]
        domain_classifier_vars = [var for var in tvars if 'domain_classifier' in var.name]
        domain_similarity = [var for var in tvars if 'domain_similarity' in var.name]

        self.pre_trainer = tf.train.AdamOptimizer(
            learning_rate=0.001
        ).minimize(self.label_loss, var_list=label_classifier_vars + feature_extractor_vars + label_classifier_controller_vars)

        self.post_trainer = tf.train.GradientDescentOptimizer(
            learning_rate=0.01
        ).minimize(self.label_loss, var_list=label_classifier_vars)

        self.domain_opt_real = tf.train.GradientDescentOptimizer(
            learning_rate=0.01
        ).minimize(self.domain_loss_real, var_list=domain_classifier_vars)

        self.domain_opt_fake = tf.train.GradientDescentOptimizer(
            learning_rate=0.01
        ).minimize(self.domain_loss_fake, var_list=feature_extractor_vars)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def calc_f1(self, probs, y):

        predicted = tf.reshape(tf.round(probs), [-1])
        TP = tf.count_nonzero(tf.multiply(predicted, y), dtype=tf.float32)
        TN = tf.count_nonzero(tf.multiply((predicted - 1), (y - 1)), dtype=tf.float32)
        FP = tf.count_nonzero(tf.multiply(predicted, (y - 1)), dtype=tf.float32)
        FN = tf.count_nonzero(tf.multiply((predicted - 1), y), dtype=tf.float32)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        # acc = (TP + TN) / (TP + TN + FP + FN)

        f1 = 2 * precision * recall / (precision + recall)

        return TP, FP, FN, precision, recall, f1

    def build_graph(self, x_0_ph, x_1_ph, x_2_ph, x_3_ph, reuse=False):
        with tf.variable_scope('layer_0_cnn', reuse=reuse):
            output_0_cnn = self.create_big_layer_cnn(tf.stack([x_0_ph, x_1_ph, x_2_ph, x_3_ph], axis=-1), reuse=reuse,
                                                     max_seq_len=self.max_seq_len, embedding_size=self.embedding_size)

        # Make classifier
        outputs = tf.layers.dense(inputs=output_0_cnn, units=self.classifier_ls, activation=tf.nn.relu, reuse=reuse,
                                  name='classifier_1')
        outputs = tf.layers.dense(inputs=outputs, units=self.classifier_ls, reuse=reuse, name='classifier_2', activation=tf.nn.relu)

        return outputs

    def create_big_layer_cnn(self, input_tensor, max_seq_len, embedding_size, reuse=False):

        # n grams
        one_grams_output_4 = self.n_gram_layer_conv_4(1, input_tensor, reuse=reuse, max_seq_len=max_seq_len,
                                                      embedding_size=embedding_size)
        two_grams_output_4 = self.n_gram_layer_conv_4(2, input_tensor, reuse=reuse, max_seq_len=max_seq_len,
                                                      embedding_size=embedding_size)
        three_grams_output_4 = self.n_gram_layer_conv_4(3, input_tensor, reuse=reuse, max_seq_len=max_seq_len,
                                                        embedding_size=embedding_size)
        four_grams_output_4 = self.n_gram_layer_conv_4(4, input_tensor, reuse=reuse, max_seq_len=max_seq_len,
                                                       embedding_size=embedding_size)
        five_grams_output_4 = self.n_gram_layer_conv_4(5, input_tensor, reuse=reuse, max_seq_len=max_seq_len,
                                                       embedding_size=embedding_size)
        six_grams_output_4 = self.n_gram_layer_conv_4(6, input_tensor, reuse=reuse, max_seq_len=max_seq_len,
                                                       embedding_size=embedding_size)

        # Concat all outputs
        all_outputs = tf.concat(
            [
                one_grams_output_4,
                two_grams_output_4,
                three_grams_output_4,
                four_grams_output_4,
                five_grams_output_4,
                six_grams_output_4
            ],
            axis=1
        )

        # Make classifier
        outputs = tf.layers.dense(inputs=all_outputs, units=self.classifier_ls * 4, activation=tf.nn.relu, reuse=reuse)

        return outputs

    def n_gram_layer_conv_4(self, size, input_tensor, max_seq_len, embedding_size, reuse=False):

        outputs = tf.layers.conv2d(input_tensor, self.cnn_ls, [size, embedding_size], reuse=reuse,
                                   name='conv_' + str(size))
        outputs = tf.layers.batch_normalization(outputs, axis=3, reuse=reuse, name='batch_norm_' + str(size))
        outputs = tf.nn.relu(outputs)

        outputs = tf.reshape(outputs, [-1, (max_seq_len - size + 1) * self.cnn_ls])

        outputs = tf.layers.dense(
            inputs=outputs, units=self.unit_ls, activation=tf.nn.relu, reuse=reuse, name='dense_after_cnn_' + str(size)
        )

        return outputs

    def train_2(
            self,
            data,
            unsup_data,
            base_path,
            early_stopping_value=50,
            save_weights=True,
            verbose=False,
            batch_size=64,
            global_best_f1=0
    ):

        trial_data = pd.read_csv('submissions_a/goldstandard.csv', encoding='latin-1')
        trial_labels = trial_data['label'].values
        del trial_data
        trial_features = pickle.load(open("bert_features_4/trial_data_features_4_char_A.pkl", "rb"))
        trial_features = [trial_features[key] for key in range(len(trial_features))]
        trial_features_0, trial_features_1, trial_features_2, trial_features_3 = self.format_batch(trial_features)
        trial_feed_dict_a = {
            self.x_0_ph: trial_features_0,
            self.x_1_ph: trial_features_1,
            self.x_2_ph: trial_features_2,
            self.x_3_ph: trial_features_3,

            self.y_ph: trial_labels
        }

        test_data = pd.read_csv('data/SubtaskB_EvaluationData_labeled.csv', header=None,
                                names=['id', 'sentence', 'label'], encoding='latin-1')
        test_labels = test_data['label'].values
        del test_data
        test_features = pickle.load(open("bert_features_4/subtask_B_test_data.pkl", "rb"))
        test_features = [test_features[key] for key in range(len(test_features))]
        test_features_0, test_features_1, test_features_2, test_features_3 = self.format_batch(test_features)
        test_feed_dict = {
            self.x_0_ph: test_features_0,
            self.x_1_ph: test_features_1,
            self.x_2_ph: test_features_2,
            self.x_3_ph: test_features_3,

            self.y_ph: test_labels
        }

        trial_data = pd.read_csv('submissions_b/goldstandard.csv', encoding='latin-1')
        trial_labels = trial_data['label'].values
        del trial_data
        trial_features = pickle.load(open("bert_features_4/trial_data_features_4_char_B.pkl", "rb"))
        trial_features = [trial_features[key] for key in range(len(trial_features))]
        trial_features_0, trial_features_1, trial_features_2, trial_features_3 = self.format_batch(trial_features)
        trial_feed_dict_b = {
            self.x_0_ph: trial_features_0,
            self.x_1_ph: trial_features_1,
            self.x_2_ph: trial_features_2,
            self.x_3_ph: trial_features_3,

            self.y_ph: trial_labels
        }

        random.shuffle(unsup_data)
        unsup_pool = cycle(unsup_data)
        def unsup_generator():
            for file in unsup_pool:
                train_data_unsup = pickle.load(open(base_path + 'hotel_reviews/' + file, 'rb'))
                dict_keys = list(train_data_unsup.keys())
                random.shuffle(dict_keys)
                for key in dict_keys:
                    yield train_data_unsup[key]
        unsup_gen = unsup_generator()

        class_0_data = []
        class_1_data = []
        class_all_data = [data[key] for key in range(len(data))]
        for key in range(len(data)):
            if data[key]['label'] == 0:
                class_0_data.append(data[key])
            elif data[key]['label'] == 1:
                class_1_data.append(data[key])

        random.shuffle(class_0_data)
        random.shuffle(class_1_data)
        random.shuffle(class_all_data)

        class_0_pool = cycle(class_0_data)
        class_1_pool = cycle(class_1_data)

        # Pretrain domain classificator
        steps = 0
        best_f1 = 0
        early_stopping_index = 0
        test_f1_scores_b = []
        val_f1_scores_b = []
        val_f1_scores_a = []
        while True:
            if steps % 2 == 0:
                train_batch = [next(class_0_pool) for _ in range(batch_size)]
            else:
                train_batch = [next(class_1_pool) for _ in range(batch_size)]

            # train_batch = [next(class_all_pool) for _ in range(batch_size)]

            # Get unsup data
            # unsup_data_batch = [next(unsup_gen) for _ in range(batch_size)]

            features_0, features_1, features_2, features_3 = self.format_batch(train_batch)
            # unsup_features_0, unsup_features_1, unsup_features_2, unsup_features_3 = self.format_batch(unsup_data_batch)

            feed_dict_label = {
                self.x_0_ph: features_0,
                self.x_1_ph: features_1,
                self.x_2_ph: features_2,
                self.x_3_ph: features_3,
                # self.x_0_ph_unsup: unsup_features_0,
                # self.x_1_ph_unsup: unsup_features_1,
                # self.x_2_ph_unsup: unsup_features_2,
                # self.x_3_ph_unsup: unsup_features_3,
                self.y_ph: [train_dat['label'] for train_dat in train_batch]
            }
            feature_extractor_loss, train_f1, _ = self.sess.run(
                [
                    self.label_loss,
                    self.f1,
                    self.pre_trainer
                ],
                feed_dict_label
            )

            trial_loss_a, trial_f1_a = self.sess.run([self.label_loss, self.f1], trial_feed_dict_a)
            trial_loss_b, trial_f1_b = self.sess.run([self.label_loss, self.f1], trial_feed_dict_b)
            test_loss_b, test_f1_b = self.sess.run([self.label_loss, self.f1], test_feed_dict)

            val_f1_scores_a.append(trial_f1_a)
            val_f1_scores_b.append(trial_f1_b)
            test_f1_scores_b.append(test_f1_b)
            print(
                'feature_extractor_loss:', '{0:.3f}'.format(feature_extractor_loss), '\t',
                'A:trial_loss:', '{0:.3f}'.format(trial_loss_a), '\t',
                'A:trial_f1:', '{0:.3f}'.format(trial_f1_a), '\t',
                'B:trial_loss:', '{0:.3f}'.format(trial_loss_b), '\t',
                'B:trial_f1:', '{0:.3f}'.format(trial_f1_b), '\t',
                steps
            )

            steps += 1

            # early stopping
            if best_f1 < trial_f1_a and trial_f1_a > 0.82:
                early_stopping_index = 0
                best_f1 = trial_f1_a

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

            if best_f1 > 0 and steps > 50:
                early_stopping_index += 1

            if early_stopping_index >= early_stopping_value:
                if verbose:
                    print("Early stopping phase 1")
                break

        val_score_a, = plt.plot(val_f1_scores_a, label='Subtask A: Validation score')
        val_score_b, = plt.plot(val_f1_scores_b, label='Subtask B: Validation score')
        test_score_b, = plt.plot(test_f1_scores_b, label='Subtask B: Test score')

        plt.legend(handles=[val_score_a, val_score_b, test_score_b])
        plt.title('Pre-training')
        plt.xlabel('train steps')
        plt.ylabel('f1-score')

        # draw vertical line from (70,100) to (70, 250)
        plt.plot([50, 50], [0.2, 0.9], 'k-', lw=2)
        plt.text(50, 0.1, 'Early Stopping')
        plt.show()
        exit(0)
        plt.figure()

        self.saver.restore(self.sess, tf.train.latest_checkpoint("best_weights"))
        print('Starting phase 2')
        steps = 0
        best_f1 = 0
        early_stopping_index = 0
        # test_f1_scores_b = []
        # val_f1_scores_b = []
        # val_f1_scores_a = []
        while True:
            if steps % 2 == 0:
                train_batch = [next(class_0_pool) for _ in range(batch_size)]
            else:
                train_batch = [next(class_1_pool) for _ in range(batch_size)]

            # Get unsup data
            unsup_data_batch = [next(unsup_gen) for _ in range(batch_size)]

            features_0, features_1, features_2, features_3 = self.format_batch(train_batch)
            unsup_features_0, unsup_features_1, unsup_features_2, unsup_features_3 = self.format_batch(unsup_data_batch)

            feed_dict_domain_real = {
                self.x_0_ph: features_0,
                self.x_1_ph: features_1,
                self.x_2_ph: features_2,
                self.x_3_ph: features_3,
                self.x_0_ph_unsup: unsup_features_0,
                self.x_1_ph_unsup: unsup_features_1,
                self.x_2_ph_unsup: unsup_features_2,
                self.x_3_ph_unsup: unsup_features_3,
            }
            domain_loss_real, _ = self.sess.run([self.domain_loss_real, self.domain_opt_real], feed_dict_domain_real)

            feed_dict_domain_fake = {
                self.x_0_ph: features_0,
                self.x_1_ph: features_1,
                self.x_2_ph: features_2,
                self.x_3_ph: features_3,
                self.x_0_ph_unsup: unsup_features_0,
                self.x_1_ph_unsup: unsup_features_1,
                self.x_2_ph_unsup: unsup_features_2,
                self.x_3_ph_unsup: unsup_features_3,
                self.y_ph: [train_dat['label'] for train_dat in train_batch]
            }
            domain_loss_fake, feature_extractor_loss, _ = self.sess.run([self.domain_loss_fake, self.label_loss, self.domain_opt_fake], feed_dict_domain_fake)

            trial_loss_a, trial_f1_a = self.sess.run([self.label_loss, self.f1], trial_feed_dict_a)
            trial_loss_b, trial_f1_b = self.sess.run([self.label_loss, self.f1], trial_feed_dict_b)
            test_loss_b, test_f1_b = self.sess.run([self.label_loss, self.f1], test_feed_dict)

            val_f1_scores_a.append(trial_f1_a)
            val_f1_scores_b.append(trial_f1_b)
            test_f1_scores_b.append(test_f1_b)
            print(
                'A:trial_loss:', '{0:.3f}'.format(trial_loss_a), '\t',
                'A:trial_f1 subtask', '{0:.3f}'.format(trial_f1_a), '\t',
                'B:trial_loss:', '{0:.3f}'.format(trial_loss_b), '\t',
                'B:trial_f1 subtask', '{0:.3f}'.format(trial_f1_b), '\t',
                steps
            )

            # early stopping
            if best_f1 < trial_f1_b and steps > 10:
                early_stopping_index = 0
                best_f1 = trial_f1_b

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

            if steps > 10:
                early_stopping_index += 1

        val_score_a, = plt.plot(val_f1_scores_a, label='Subtask A: Validation score')
        val_score_b, = plt.plot(val_f1_scores_b, label='Subtask B: Validation score')
        test_score_b, = plt.plot(test_f1_scores_b, label='Subtask B: Test score')

        plt.legend(handles=[val_score_a, val_score_b, test_score_b])
        plt.title('Adversarial training')
        plt.xlabel('train steps')
        plt.ylabel('f1-score')
        plt.draw()
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
            }

            batch_probs = self.sess.run(self.label_probs, feed_dict=feed_dict)
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