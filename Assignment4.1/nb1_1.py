import argparse
import math
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict
import numpy as np
import pandas as pd
import time

class BernoulliNaiveBayes:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_log_prob_neg_ = None
        self.classes_ = None
        self.stemmer = PorterStemmer()
        self.vocabulary = {}
        self.feature_count = 0
        self.class_counts = None  # Added to store class counts

    def preprocess(self, text, stop_words):
        text = text.lower()
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return set(tokens)

    def train(self, train_file, stop_words):
        df_train = pd.read_csv(train_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        df_train = df_train.dropna(subset=[1, 2])

        labels = df_train[1].values
        texts = df_train[2].values

        # Preprocess texts
        df_train['features'] = df_train[2].apply(lambda text: self.preprocess(text, stop_words))

        # Build vocabulary
        all_features = set()
        for features in df_train['features']:
            all_features.update(features)
        self.vocabulary = {feature: idx for idx, feature in enumerate(sorted(all_features))}
        self.feature_count = len(self.vocabulary)

        # Build document-term matrix
        num_docs = len(df_train)
        num_features = self.feature_count
        X = np.zeros((num_docs, num_features), dtype=np.int8)

        for i, features in enumerate(df_train['features']):
            indices = [self.vocabulary[feature] for feature in features if feature in self.vocabulary]
            X[i, indices] = 1

        # Map labels to indices
        self.classes_ = np.unique(labels)
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y = np.array([class_to_index[label] for label in labels])
        num_classes = len(self.classes_)

        # Compute class prior probabilities
        class_counts = np.bincount(y)
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.class_counts = class_counts  # Store class counts

        # Compute feature counts per class
        feature_counts = np.zeros((num_classes, num_features), dtype=np.int32)
        for c in range(num_classes):
            X_c = X[y == c]
            feature_counts[c] = X_c.sum(axis=0)

        # Compute feature probabilities with Laplace smoothing
        self.feature_log_prob_ = np.log((feature_counts + 1) / (class_counts[:, None] + 2))
        self.feature_log_prob_neg_ = np.log(1 - np.exp(self.feature_log_prob_))

        return num_docs

    def predict(self, X_test_features_list):
        num_docs = len(X_test_features_list)
        num_features = self.feature_count

        # Initialize document-term matrix and unseen feature flags
        X = np.zeros((num_docs, num_features), dtype=np.int8)
        has_unseen_features = np.zeros(num_docs, dtype=bool)

        # Build document-term matrix and identify unseen features
        for i, features in enumerate(X_test_features_list):
            feature_set = set(features)
            known_features = feature_set & self.vocabulary.keys()
            unseen_features = feature_set - self.vocabulary.keys()

            # Map known features to indices
            indices = [self.vocabulary[feature] for feature in known_features]
            X[i, indices] = 1

            # Mark if there are unseen features
            if unseen_features:
                has_unseen_features[i] = True

        # Compute joint log likelihood
        jll = X @ self.feature_log_prob_.T
        jll += (1 - X) @ self.feature_log_prob_neg_.T
        jll += self.class_log_prior_

        # Compute unseen feature penalty per class
        unseen_prob = np.log(1 / (self.class_counts + 2))  # Shape: (num_classes,)

        # Adjust joint log likelihood with unseen feature penalty
        jll[has_unseen_features] += unseen_prob 

        # Predict class with highest joint log likelihood
        indices = np.argmax(jll, axis=1)
        predicted_labels = [self.classes_[idx] for idx in indices]
        return predicted_labels

    def test(self, test_file, stop_words):
        df_test = pd.read_csv(test_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        df_test = df_test.dropna(subset=[2])

        # Preprocess texts
        df_test['features'] = df_test[2].apply(lambda text: self.preprocess(text, stop_words))
        test_features_list = df_test['features'].tolist()

        # Make predictions
        predictions = self.predict(test_features_list)
        predictions_labels = predictions

        return predictions_labels

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f)

def main():
    s_t = time.time()
    parser = argparse.ArgumentParser(description='Bernoulli Naive Bayes for Fake News Detection')
    parser.add_argument('--train', required=False, help='Path to the training file', default='train.tsv')
    parser.add_argument('--test', required=False, help='Path to the test file', default='valid.tsv')
    parser.add_argument('--out', required=False, help='Path to the output file', default='output_1.txt')
    parser.add_argument('--stop', required=False, help='Path to stopwords file', default='stopwords.txt')
    args = parser.parse_args()

    stop_words = load_stopwords(args.stop)

    classifier = BernoulliNaiveBayes()
    rows = classifier.train(args.train, stop_words)
    print(f'Rows discovered: {rows}')
    predictions = classifier.test(args.test, stop_words)

    with open(args.out, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    e_t = time.time()
    print(f'Time taken: {e_t - s_t}')

if __name__ == "__main__":
    main()
