import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from SubtaskB.model.model import nn

def predict():
    trial_features = pickle.load(open("bert_features_4/trial_data_features_4_char_B.pkl", "rb"))
    nn_1 = nn(40, 768, 64)

    trial_data = pd.read_csv('data/SubtaskB_Trial_Test.csv', header=None, names=['id', 'text', 'label'],
                             encoding='latin-1')
    probs = nn_1.predict(trial_features)

    trial_data['label'] = np.reshape(probs.astype(int), -1)
    trial_data.to_csv('submissions_b/submission.csv', header=False, index=False)


def predict_test_data():
    features = pickle.load(open("eval_data/subtask_B.pkl", "rb"))
    nn_1 = nn(40, 768, 64)

    data = pd.read_csv('eval_data/SubtaskB_EvaluationData.csv', header=None, names=['id', 'text', 'label'],
                             encoding='latin-1')
    probs = nn_1.predict(features)

    data['label'] = np.reshape(probs.astype(int), -1)
    data.to_csv('eval_data/submission.csv', header=False, index=False)


def train():
    import os
    base_path = 'C:/Users/tcaba/PycharmProjects/bert-master/data/'
    train_features = pickle.load(open(base_path + "v1.4/train_data_features_4v1.4.pkl", "rb"))
    unsup_feature_files = [file for file in os.listdir(base_path + 'hotel_reviews') if '.txt' not in file]

    nn_1 = nn(40, 768, 64)
    nn_1.train_2(train_features, unsup_feature_files, base_path, verbose=True)

def try_out():
    import shutil, os
    base_path = 'C:/Users/tcaba/PycharmProjects/bert-master/data/'
    train_features = pickle.load(open(base_path + "v1.4/train_data_features_4v1.4.pkl", "rb"))
    unsup_feature_files = [file for file in os.listdir(base_path + 'hotel_reviews') if '.txt' not in file]

    nn_1 = nn(40, 768, 64)

    best_f1 = 0.84
    while True:

        try:
            val = nn_1.train_2(train_features, unsup_feature_files, base_path, verbose=True, global_best_f1=best_f1)
        except:
            nn_1.sess.run(tf.global_variables_initializer())
            continue

        if val > best_f1:
            dest_folder = 'weights_eval/val_' + str(val)
            shutil.move('best_weights', dest_folder)
            os.makedirs('best_weights')
            best_f1 = val

            print(val)

        nn_1.sess.run(tf.global_variables_initializer())


if __name__ == "__main__":
    train()