import argparse
import csv
import math
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict
import numpy as np
import pandas as pd

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        self.stemmer = PorterStemmer()
        self.class_mapping = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        self.total_words_in_class = defaultdict(int)

    def preprocess(self, text, stop_words):
        text = text.lower()
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens  # Return list, not set, because frequency matters in Multinomial NB

    def train(self, train_file, stop_words):
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        total_docs = 0
        
        df_train = pd.read_csv(train_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        
        rows = len(df_train)
        print(f"Total rows read: {rows}")
        
        for index, row in df_train.iterrows():
            if len(row) < 3:
                continue
            label, text = row[1], row[2]
            features = self.preprocess(text, stop_words)
            self.vocabulary.update(features)
            
            class_counts[label] += 1
            total_docs += 1
            
            for feature in features:
                feature_counts[label][feature] += 1
                self.total_words_in_class[label] += 1

        for label in class_counts:
            self.class_probs[label] = math.log(class_counts[label] / total_docs)
            for feature in self.vocabulary:
                count = feature_counts[label][feature]
                self.feature_probs[label][feature] = math.log((count + 1) / (self.total_words_in_class[label] + len(self.vocabulary)))
        
        return rows

    def test(self, test_file, stop_words):
        predictions = []

        df_test = pd.read_csv(test_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        
        for index, row in df_test.iterrows():
            if len(row) < 3:
                continue
            text = row[2]
            features = self.preprocess(text, stop_words)
            prediction = self.predict(features)
            predictions.append(prediction)
            
        return predictions

    def predict(self, features):
        best_label = None
        best_score = float('-inf')

        for label in self.class_probs:
            score = self.class_probs[label]
            unseen_prob = math.log(1 / (self.total_words_in_class[label] + len(self.vocabulary)))
            
            for feature in features:
                if feature in self.vocabulary:
                    score += self.feature_probs[label][feature]
                else:
                    score += unseen_prob  # Apply Laplace smoothing for each unseen word
            
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f)

def main():
    parser = argparse.ArgumentParser(description='Multinomial Naive Bayes for Fake News Detection')
    parser.add_argument('--train', required=False, help='Path to the training file', default='train.tsv')
    parser.add_argument('--test', required=False, help='Path to the test file', default='valid.tsv')
    parser.add_argument('--out', required=False, help='Path to the output file', default='output_2.txt')
    parser.add_argument('--stop', required=False, help='Path to stopwords file', default='stopwords.txt')
    args = parser.parse_args()

    stop_words = load_stopwords(args.stop)

    classifier = MultinomialNaiveBayes()
    rows = classifier.train(args.train, stop_words)
    print(f'Rows discovered: {rows}')
    predictions = classifier.test(args.test, stop_words)

    with open(args.out, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

if __name__ == "__main__":
    main()
