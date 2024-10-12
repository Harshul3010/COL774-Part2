import argparse
import csv
import math
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.util import ngrams
import numpy as np
import pandas as pd

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        self.stemmer = PorterStemmer()
        self.class_mapping = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']

    def preprocess(self, text, stop_words):
        text = text.lower()
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Generate uni-grams and bi-grams
        uni_grams = set(tokens)
        bi_grams = set(ngrams(tokens, 2))
        all_grams = uni_grams.union(bi_grams)

        return all_grams

    def train(self, train_file, stop_words):
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        total_docs = 0
        
        # Read data using pandas to prevent row drops
        df_train = pd.read_csv(train_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        
        rows = len(df_train)
        print(f"Total rows read: {rows}")
        
        for index, row in df_train.iterrows():
            if len(row) < 3:  # Skip rows with less than 3 columns
                continue
            label, text = row[1], row[2]
            features = self.preprocess(text, stop_words)
            print(f"Features for document {index + 1}: {features}")
            self.vocabulary.update(features)
            
            class_counts[label] += 1
            total_docs += 1
            
            for feature in features:
                feature_counts[label][feature] += 1
        

        for label in class_counts:
            self.class_probs[label] = math.log(class_counts[label] / total_docs)  # Prior probability of the class
            total_words_in_class = sum(feature_counts[label].values())  # Total words in this class

            for feature in self.vocabulary:
                count = feature_counts[label][feature]
                # Apply Laplace smoothing with c=1 for Multinomial Naive Bayes
                self.feature_probs[label][feature] = math.log((count + 1) / (total_words_in_class + len(self.vocabulary)))

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
            for feature in features:
                if feature in self.vocabulary:
                    score += self.feature_probs[label][feature]

            if score > best_score:
                best_score = score
                best_label = label

        return best_label

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f)

def main():
    parser = argparse.ArgumentParser(description='Multinomial Naive Bayes with Uni-grams and Bi-grams')
    parser.add_argument('--train', required=False, help='Path to the training file', default='train.tsv')
    parser.add_argument('--test', required=False, help='Path to the test file', default='valid.tsv')
    parser.add_argument('--out', required=False, help='Path to the output file', default='output_3b.txt')
    parser.add_argument('--stop', required=False, help='Path to stopwords file', default='stopwords.txt')
    args = parser.parse_args()

    stop_words = load_stopwords(args.stop)

    classifier = MultinomialNaiveBayes()
    classifier.train(args.train, stop_words)
    predictions, accuracy = classifier.test(args.test, stop_words)

    with open(args.out, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
