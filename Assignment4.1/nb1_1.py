import argparse
import csv
import math
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict
import numpy as np
import pandas as pd

class BernoulliNaiveBayes:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        self.class_counts = defaultdict(int)
        self.vocabulary = set()
        self.stemmer = PorterStemmer()
        self.class_mapping = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']

    def preprocess(self, text, stop_words):
        text = text.lower()  # Convert to lowercase
        tokens = text.split()  # Split on whitespace
        # print(f"Original lowercase tokens: {tokens}")
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]
        # print(f"After stopword removal: {tokens}")
        
        # Apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        # print(f"After stemming: {tokens}")
        
        return set(tokens)


    def train(self, train_file, stop_words):
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
            
            self.class_counts[label] += 1
            total_docs += 1
            
            for feature in features:
                feature_counts[label][feature] += 1

        for label in self.class_counts:
            self.class_probs[label] = math.log(self.class_counts[label] / total_docs)
            for feature in self.vocabulary:
                count = feature_counts[label][feature]
                self.feature_probs[label][feature] = math.log((count + 1) / (self.class_counts[label] + 2))
        
        return rows


    def test(self, test_file, stop_words):
        correct = 0
        total = 0
        predictions = []

        # Read data using pandas to prevent row drops
        df_test = pd.read_csv(test_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        
        # Iterate over each row in the test data
        for index, row in df_test.iterrows():
            if len(row) < 3:  # Skip rows with less than 3 columns
                continue
            label, text = row[1], row[2]
            features = self.preprocess(text, stop_words)
            prediction = self.predict(features)
            predictions.append(prediction)
            
            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0  # Handle division by zero in case total is zero
        return predictions, accuracy

    def predict(self, features):
        best_label = None
        best_score = float('-inf')
        
        for label in self.class_probs:
            score = self.class_probs[label]
            for feature in self.vocabulary:
                if feature in features:
                    score += self.feature_probs[label][feature]
                else:
                    score += math.log(1 - math.exp(self.feature_probs[label][feature]))
            
            # Handle unseen words
            unseen_words = features - self.vocabulary
            if unseen_words:
                unseen_prob = math.log(1 / (2 + self.class_counts[label]))
                score +=  unseen_prob
            
            if score > best_score:
                best_score = score
                best_label = label
        
        return best_label


def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f)

def main():
    parser = argparse.ArgumentParser(description='Bernoulli Naive Bayes for Fake News Detection')
    parser.add_argument('--train', required=False, help='Path to the training file', default='train.tsv')
    parser.add_argument('--test', required=False, help='Path to the test file', default='valid.tsv')
    parser.add_argument('--out', required=False, help='Path to the output file', default='output_1.txt')
    parser.add_argument('--stop', required=False, help='Path to stopwords file', default='stopwords.txt')
    args = parser.parse_args()

    stop_words = load_stopwords(args.stop)

    classifier = BernoulliNaiveBayes()
    rows=classifier.train(args.train, stop_words)
    print(f'rows discovered: {rows}')
    predictions, accuracy = classifier.test(args.test, stop_words)

    with open(args.out, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
