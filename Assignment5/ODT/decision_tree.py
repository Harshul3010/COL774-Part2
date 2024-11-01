import numpy as np
import pandas as pd
from logistic_regression import logistic_regression
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
    def __init__(self, max_depth, min_samples_split=2):
        self.max_depth = max_depth
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
        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]
        sorted_y = y[sorted_indices]

        best_threshold = None
        best_gini = float('inf')

        for i in range(1, len(sorted_projections)):
            threshold = (sorted_projections[i - 1] + sorted_projections[i]) / 2
            left_mask = sorted_projections <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
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
            node.prediction = np.argmax(np.bincount(y))
            print(f"Creating leaf node with prediction: {node.prediction}")
            return node

        try:
            print("Running logistic regression to find weights...")
            node.weights = logistic_regression(X, y)
            projections = X @ node.weights
            node.threshold = self._find_best_threshold(projections, y)
            
            if node.threshold is None:
                raise Exception("No valid threshold found")
            
            left_mask = projections <= node.threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                raise Exception("Insufficient samples in split")

            node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, 2 * node_id)
            node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1, 2 * node_id + 1)
            
        except Exception as e:
            print(f"Exception encountered: {e}. Creating a leaf node with prediction.")
            node.leaf = True
            node.prediction = np.argmax(np.bincount(y))
        
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
                if y.size == 0: 
                    print(f"Skipping pruning at node {node.node_id} due to empty labels.")
                    return node

                original_accuracy = validate_node_accuracy(node, X, y)
                print(f"Original accuracy at node {node.node_id}: {original_accuracy:.4f}")

                leaf_prediction = np.argmax(np.bincount(y)) 
                node.leaf = True
                node.prediction = leaf_prediction
                pruned_accuracy = validate_node_accuracy(node, X, y)
                print(f"Pruned accuracy at node {node.node_id} if set as leaf: {pruned_accuracy:.4f}")

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
    if len(sys.argv) < 2:
        print("Please provide command and required arguments")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "train":
        if len(sys.argv) < 3:
            print("Please specify unpruned or pruned mode")
            sys.exit(1)
            
        mode = sys.argv[2]
        
        if mode == "unpruned":
            if len(sys.argv) != 6:
                print("Usage: python decision_tree.py train unpruned train.csv max_depth weight.csv")
                sys.exit(1)
                
            train_file = sys.argv[3]
            max_depth = int(sys.argv[4])
            weight_file = sys.argv[5]
            
            X_train, y_train = process_data(train_file)
            tree = ObliqueDecisionTree(max_depth=max_depth, min_samples_split=2)
            tree.fit(X_train, y_train)
            tree.save_weights(weight_file)

        elif mode == "pruned":
            if len(sys.argv) != 7:
                print("Usage: python decision_tree.py train pruned train.csv val.csv max_depth weight.csv")
                sys.exit(1)
                
            train_file = sys.argv[3]
            val_file = sys.argv[4]
            max_depth = int(sys.argv[5])
            weight_file = sys.argv[6]
            
            X_train, y_train = process_data(train_file)
            X_val, y_val = process_data(val_file)
            
            tree = ObliqueDecisionTree(max_depth=max_depth)
            tree.fit(X_train, y_train)
            tree.prune(X_val, y_val)
            tree.save_weights(weight_file)
            
        else:
            print("Invalid mode. Use 'unpruned' or 'pruned'")
            sys.exit(1)
            
    elif command == "test":
        if len(sys.argv) != 7:
            print("Usage: python decision_tree.py test train.csv val.csv test.csv max_depth prediction.csv")
            sys.exit(1)
            
        train_file = sys.argv[2]
        val_file = sys.argv[3]
        test_file = sys.argv[4]
        max_depth = int(sys.argv[5])
        pred_file = sys.argv[6]
        
        X_train, y_train = process_data(train_file)
        X_val, y_val = process_data(val_file)
        test_data = pd.read_csv(test_file)
        X_test = test_data.iloc[:, :-1].values
        
        tree = ObliqueDecisionTree(max_depth=max_depth)
        tree.fit(X_train, y_train)
        tree.save_weights("weights_real_unpruned.csv")
        tree.prune(X_val, y_val)
        tree.save_weights("weights_real_pruned.csv")
        
        predictions = tree.predict(X_test)
        pd.DataFrame(predictions).to_csv(pred_file, header=False, index=False)
        print("Predictions saved successfully.")
