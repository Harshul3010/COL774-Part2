import argparse
import csv
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.util import ngrams
import pandas as pd
import string

class CustomLabelEncoder:
    def __init__(self):
        self.encoding = {}
        self.decoding = {}
        self.next_code = 0

    def fit(self, values):
        unique_values = set()
        for value in values:
            if isinstance(value, str):
                # Split comma-separated values
                for sub_value in value.split(','):
                    unique_values.add(sub_value.strip())
            else:
                unique_values.add(value)

        for value in unique_values:
            if value not in self.encoding:
                self.encoding[value] = self.next_code
                self.decoding[self.next_code] = value
                self.next_code += 1

    def transform(self, values):
        transformed = []
        for value in values:
            if isinstance(value, str):
                # Handle comma-separated values
                encoded = [self.encoding.get(sub_value.strip(), -1) for sub_value in value.split(',')]
                transformed.append(encoded)
            else:
                transformed.append([self.encoding.get(value, -1)])
        return transformed

    def inverse_transform(self, codes):
        return [[self.decoding.get(code, None) for code in code_list] for code_list in codes]

class EnhancedMultinomialNaiveBayes:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        self.stemmer = PorterStemmer()
        self.class_mapping = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        self.total_words_in_class = defaultdict(int)
        self.categorical_features = ['subject', 'speaker', 'job_title', 'state', 'party']
        self.numerical_features = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_fire_counts']
        self.label_encoders = {feature: CustomLabelEncoder() for feature in self.categorical_features}

    def preprocess(self, text, stop_words):
        text = str(text).lower()  # Convert to string in case of NaN
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]

        uni_grams = tokens
        # bi_grams = [' '.join(gram) for gram in ngrams(tokens, 2)]
        # tri_grams = [' '.join(gram) for gram in ngrams(tokens, 3)]

        # Combine all n-grams into one list
        all_grams = uni_grams 

        return all_grams

    def train(self, train_file, stop_words):
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        total_docs = 0
        
        df_train = pd.read_csv(train_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        
        rows = len(df_train)
        print(f"Total rows read: {rows}")

        # Fit label encoders
        for feature in self.categorical_features:
            self.label_encoders[feature].fit(df_train[df_train.columns[3 + self.categorical_features.index(feature)]].fillna('Unknown'))
        
        for index, row in df_train.iterrows():
            a = str(row[2]) + " " + str(row[13])+ " " + str(row[7])
            label, text = row[1], a
            features = self.preprocess(text, stop_words)
            
            self.vocabulary.update(features)
            
            class_counts[label] += 1
            total_docs += 1
            
            for feature in features:
                feature_counts[label][feature] += 1
                self.total_words_in_class[label] += 1

            # Add encoded categorical features
            for i, feature in enumerate(self.categorical_features):
                if i+3 < len(row):
                    encoded_values = self.label_encoders[feature].transform([str(row[i+3])])[0]
                    for encoded_value in encoded_values:
                        feature_value = f"{feature}_{encoded_value}"
                        feature_counts[label][feature_value] += 1
                        self.vocabulary.add(feature_value)

            # Add numerical features directly
            for i, feature in enumerate(self.numerical_features):
                if i+8 < len(row):
                    feature_value = f"{feature}_{row[i+8]}"
                    feature_counts[label][feature_value] += 1
                    self.vocabulary.add(feature_value)

        for label in class_counts:
            self.class_probs[label] = math.log(class_counts[label] / total_docs)
            for feature in self.vocabulary:
                count = feature_counts[label][feature]
                self.feature_probs[label][feature] = math.log((count + 1) / (self.total_words_in_class[label] + len(self.vocabulary)))
        
        print(f"Number of classes: {len(class_counts)}")
        return rows

    def test(self, test_file, stop_words):
        correct = 0
        total = 0
        predictions = []

        df_test = pd.read_csv(test_file, sep="\t", header=None, quoting=3, encoding='utf-8')
        
        for index, row in df_test.iterrows():
            a = str(row[2]) + " " + str(row[13]) + " " + str(row[7])
            label, text = row[1], a
            features = self.preprocess(text, stop_words)
            
            # Add encoded categorical features
            for i, feature in enumerate(self.categorical_features):
                if i+3 < len(row):
                    encoded_values = self.label_encoders[feature].transform([str(row[i+3])])[0]
                    for encoded_value in encoded_values:
                        features.append(f"{feature}_{encoded_value}")

            # Add numerical features directly
            for i, feature in enumerate(self.numerical_features):
                if i+8 < len(row):
                    features.append(f"{feature}_{row[i+8]}")
            
            prediction = self.predict(features)
            predictions.append(prediction)
            
            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        return predictions, accuracy, correct, total

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
    parser = argparse.ArgumentParser(description='Enhanced Multinomial Naive Bayes for Fake News Detection')
    parser.add_argument('--train', required=False, help='Path to the training file', default='train.tsv')
    parser.add_argument('--test', required=False, help='Path to the test file', default='valid.tsv')
    parser.add_argument('--out', required=False, help='Path to the output file', default='output_p2.txt')
    parser.add_argument('--stop', required=False, help='Path to stopwords file', default='stopwords.txt')
    args = parser.parse_args()

    stop_words = load_stopwords(args.stop)

    classifier = EnhancedMultinomialNaiveBayes()
    rows = classifier.train(args.train, stop_words)
    print(f'Rows discovered: {rows}')
    predictions, accuracy, correct, total = classifier.test(args.test, stop_words)

    print(f"Writing {len(predictions)} predictions to {args.out}")
    with open(args.out, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {correct} out of {total}")

if __name__ == "__main__":
    main()