import argparse
import csv
import math
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.util import ngrams
import numpy as np

class BernoulliNaiveBayes:
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

        with open(train_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3:
                    continue
                label, text = row[1], row[2]
                features = self.preprocess(text, stop_words)
                self.vocabulary.update(features)
                
                class_counts[label] += 1
                total_docs += 1

                for feature in features:
                    feature_counts[label][feature] += 1

        for label in class_counts:
            self.class_probs[label] = math.log(class_counts[label] / total_docs)
            for feature in self.vocabulary:
                count = feature_counts[label][feature]
                total = class_counts[label]
                self.feature_probs[label][feature] = math.log((count + 1) / (total + 2))

    def test(self, test_file, stop_words):
        correct = 0
        total = 0
        predictions = []

        with open(test_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3:
                    continue
                label, text = row[1], row[2]
                features = self.preprocess(text, stop_words)
                prediction = self.predict(features)
                predictions.append(prediction)

                if prediction == label:
                    correct += 1
                total += 1

        accuracy = correct / total
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
                    exp_prob = math.exp(self.feature_probs[label][feature])
                    score += math.log(1 - exp_prob) if exp_prob < 1 else float('-inf')

            if score > best_score:
                best_score = score
                best_label = label

        return best_label

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f)

def main():
    parser = argparse.ArgumentParser(description='Bernoulli Naive Bayes with Uni-grams and Bi-grams')
    parser.add_argument('--train', required=True, help='Path to the training file')
    parser.add_argument('--test', required=True, help='Path to the test file')
    parser.add_argument('--out', required=True, help='Path to the output file')
    parser.add_argument('--stop', required=True, help='Path to stopwords file')
    args = parser.parse_args()

    stop_words = load_stopwords(args.stop)

    classifier = BernoulliNaiveBayes()
    classifier.train(args.train, stop_words)
    predictions, accuracy = classifier.test(args.test, stop_words)

    with open(args.out, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
