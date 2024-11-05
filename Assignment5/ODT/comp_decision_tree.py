# allowed imports sys, os, math, numpy, pandas, csv, json, sklearn, cvxpy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

class Node:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.depth = depth
        self.weights = None
        self.threshold = None
        self.leaf = False
        self.prediction = None
        self.node_id = None
        self.feature_count = None

class ObliqueDecisionTree:
    def __init__(self, max_depth, min_samples_split=2, regularization = 0.01):
        self.max_depth = max_depth
        self.regularization = regularization
        self.min_samples_split = min_samples_split
        self.tree = None
        self.node_count = 0
        self.feature_count = None

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _find_best_threshold(self, projections, y):
        print("Finding the best threshold...")

        if np.any(np.isnan(projections)) or np.any(np.isinf(projections)):
            raise Exception("Projections contain invalid values")

        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]
        sorted_y = y[sorted_indices]

        best_threshold = None
        best_gini = np.inf

        for i in range(1, len(sorted_projections)):
            threshold = (sorted_projections[i - 1] + sorted_projections[i]) / 2
            left_mask = sorted_projections <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < self.min_samples_split and np.sum(right_mask) < self.min_samples_split:
                continue
            
            left_gini = self._gini_impurity(sorted_y[left_mask])
            right_gini = self._gini_impurity(sorted_y[right_mask])
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            gini = (n_left * left_gini + n_right * right_gini) / len(y)

            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold
            
        print(f"Best threshold found: {best_threshold} with Gini impurity: {best_gini}")
        return best_threshold

    def _should_split(self, y, depth):
        if depth >= self.max_depth:
            return False
        if len(y) < self.min_samples_split:
            return False
        if len(np.unique(y)) == 1:
            return False
        return True

    def _build_tree(self, X, y, depth=0, node_id=1):
        print(f"Building tree at depth {depth}, node ID {node_id}...")
        node = Node(depth)
        node.node_id = node_id
        node.feature_count = X.shape[1]
        self.node_count = max(self.node_count, node_id)

        if not self._should_split(y, depth):
            node.leaf = True
            counts = np.bincount(y, minlength=2)
            if counts[0] >= counts[1]:
                node.prediction = 0
            else:
                node.prediction = 1
            print(f"Creating leaf node with prediction: {node.prediction}")
            return node

        try:
            print("Running logistic regression to find weights...")
            model = LogisticRegression(C=1/self.regularization, penalty='l2', solver='liblinear', fit_intercept=False)
            model.fit(X, y)
            node.weights = model.coef_.flatten()

            if np.any(np.isnan(node.weights)) or np.any(np.isinf(node.weights)):
                raise Exception("Logistic regression produced invalid weights")

            projections = X @ node.weights

            if np.any(np.isnan(projections)) or np.any(np.isinf(projections)):
                raise Exception("Projections contain invalid values")

            node.threshold = self._find_best_threshold(projections, y)
            
            if node.threshold is None or np.isnan(node.threshold) or np.isinf(node.threshold):
                raise Exception("No valid threshold found")
            
            left_mask = projections <= node.threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_split and np.sum(right_mask) < self.min_samples_split:
                raise Exception("Insufficient samples in split")

            node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, 2 * node_id)
            node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1, 2 * node_id + 1)
            
        except Exception as e:
            print(f"Exception encountered: {e}. Creating a leaf node with prediction.")
            node.leaf = True
            counts = np.bincount(y, minlength=2)
            if counts[0] >= counts[1]:
                node.prediction = 0
            else:
                node.prediction = 1

        return node

    def fit(self, X, y):
        print("Starting training of the decision tree...")
        self.feature_count = X.shape[1]
        self.node_count = 0
        self.tree = self._build_tree(X, y)
        print("Training completed.")
        return self

    def _predict_single(self, x, node):
        if node.leaf:
            return node.prediction
        
        projection = np.dot(x, node.weights)
        if projection <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        print("Making predictions...")
        return np.array([self._predict_single(x, self.tree) for x in X])

    def save_weights(self, filename):
        print(f"Saving weights to {filename}...")
        weights_data = []
        
        def traverse(node):
            if node is None or node.leaf:
                return
            row = [node.node_id] + list(node.weights) + [node.threshold]
            weights_data.append(row)
            traverse(node.left)
            traverse(node.right)
        
        traverse(self.tree)
        df = pd.DataFrame(weights_data)
        df.to_csv(filename, header=False, index=False)
        print("Weights saved successfully.")

    def prune(self, X_val, y_val):
        def validate_node_accuracy(node, X, y):
            if y.size == 0:
                return 0
            predictions = [self._predict_single(x, node) for x in X]
            return np.mean(np.array(predictions) == y)

        def prune_subtree(node, X, y):
            if node is None or node.leaf:
                return node

            print(f"Pruning subtree at node ID {node.node_id}...")

            left_mask = X @ node.weights <= node.threshold
            right_mask = ~left_mask
            node.left = prune_subtree(node.left, X[left_mask], y[left_mask])
            node.right = prune_subtree(node.right, X[right_mask], y[right_mask])

            if node.left and node.left.leaf and node.right and node.right.leaf:
                original_accuracy = validate_node_accuracy(node, X, y)
                counts = np.bincount(y, minlength=2)
                leaf_prediction = np.argmax(counts)

                node.leaf = True
                node.prediction = leaf_prediction
                pruned_accuracy = validate_node_accuracy(node, X, y)

                if pruned_accuracy >= original_accuracy:
                    print(f"Node {node.node_id} pruned to leaf with prediction {leaf_prediction}")
                    return node

                node.leaf = False
                node.prediction = None
                print(f"Node {node.node_id} restored to internal node")

            return node
        
        print("Starting bottom-up pruning...")
        self.tree = prune_subtree(self.tree, X_val, y_val)
        print("Pruning completed.")
        return self        

def process_data(filename):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename) #Write header = None for a and b part
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)
    print("Data loaded successfully.")
    return X, y

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python comp_decision_tree.py test train.csv val.csv test.csv prediction.csv weight.csv")
        sys.exit(1)
        
    train_file = sys.argv[2]
    val_file = sys.argv[3]
    test_file = sys.argv[4]
    pred_file = sys.argv[5]
    weight_file = sys.argv[6]
    
    X_train, y_train = process_data(train_file)
    X_val, y_val = process_data(val_file)
    X_test, _ = process_data(test_file)

    tree = ObliqueDecisionTree(max_depth=10, min_samples_split=15, regularization=0.01)
    tree.fit(X_train, y_train)
    tree.prune(X_val, y_val)
    tree.save_weights(weight_file)

    predictions = tree.predict(X_test)
    pd.DataFrame(predictions).to_csv(pred_file, header=False, index=False)
