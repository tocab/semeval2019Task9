import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from SubtaskA.model.model import nn


def predict():
    # trial_features = pickle.load(open("bert_features_4/trial_data_features_4_char_B.pkl", "rb"))
    trial_features = pickle.load(open("bert_features_4/trial_data_features_4_char_A.pkl", "rb"))
    nn_1 = nn(40, 768, 64)

    # trial_data = pd.read_csv('data/SubtaskB_Trial_Test.csv', header=None, names=['id', 'text', 'label'], encoding='latin-1')
    trial_data = pd.read_csv('data/TrialData_SubtaskA_Test.csv', header=None, names=['id', 'text', 'label'],
                             encoding='latin-1')
    probs = nn_1.predict(trial_features)

    trial_data['label'] = np.reshape(probs.astype(int), -1)
    trial_data.to_csv('submissions/submission.csv', header=False, index=False)


def predict_eval_data():
    # trial_features = pickle.load(open("bert_features_4/trial_data_features_4_char_B.pkl", "rb"))
    features = pickle.load(open("eval_data/subtask_A.pkl", "rb"))
    nn_1 = nn(40, 768, 64)

    data = pd.read_csv('eval_data/SubtaskA_EvaluationData.csv', header=None, names=['id', 'text', 'label'],
                       encoding='latin-1')
    probs = nn_1.predict(features)

    data['label'] = np.reshape(probs.astype(int), -1)
    data.to_csv('eval_data/submission.csv', header=False, index=False)


def train():
    train_features = pickle.load(open("bert_features_4/v1.4/train_data_features_4v1.4.pkl", "rb"))

    nn_1 = nn(40, 768, 64)
    nn_1.train(train_features, verbose=True)


def try_out():
    import shutil, os
    train_features = pickle.load(open("bert_features_4/v1.4/train_data_features_4v1.4.pkl", "rb"))
    nn_1 = nn(40, 768, 64)

    best_f1 = 0
    while True:

        try:
            val = nn_1.train(train_features, verbose=True, global_best_f1=best_f1)
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
