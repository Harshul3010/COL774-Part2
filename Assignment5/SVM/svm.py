import numpy as np
import pandas as pd
import cvxpy as cp
import json
import sys
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    print(f"Loading data from {filename}")
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print("Data loaded successfully.")
    # Converting labels from {0, 1} to {-1, 1}
    y = 2 * y - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

def train_svm(X, y, C=1.0, tolerance=1e-4):

    print("Starting SVM training...")
    n_samples, n_features = X.shape
    print("n_sample - ", n_samples)
    print("n_features - ", n_features)
    
    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    ei = cp.Variable(n_samples)  

    print("w-", w)
    print("b - ", b)
    print("ei-", ei)

    objective = cp.Minimize(0.5 * cp.norm(w, 1) + C * cp.sum(ei))

    constraints = [
        y[i] * (X[i, :] @ w + b) >= 1 - ei[i] for i in range(n_samples)
    ]
    constraints += [ei[i] >= 0 for i in range(n_samples)]  # Slack variables should be non-negative

    problem = cp.Problem(objective, constraints)
    try:
        print("Solving the problem using solver...")
        problem.solve()
        
        print("Solver status:", problem.status)
        if problem.status != 'optimal':
            print("Solver did not find an optimal solution.")
            return None, None, False
        
        w_opt = w.value
        b_opt = b.value
        slack_values = ei.value
        print("Optimal weights:", w_opt)
        print("Optimal bias:", b_opt)
        print("Slack values:", slack_values)
        
        # Determine if linearly separable by checking slack variables
        is_separable = np.all(slack_values < tolerance)
        print("Is data linearly separable?", is_separable)
        
        return w_opt, b_opt, is_separable
        
    except Exception as e:
        print(f"Solver error: {e}")
        return None, None, False

def find_support_vectors(X, y, w, b, tolerance=1e-4):
    print("Finding support vectors...")
    margins = y * (X @ w + b)
    print("Margins calculated:", margins)
    
    sv_indices = np.where((margins >= 1 - tolerance) & (margins <= 1 + tolerance))[0]
    print("Support vector indices:", sv_indices)
    
    return sorted(sv_indices.tolist())

def save_results(weights, bias, is_separable, support_vectors, weight_file, sv_file):
    print(f"Saving results to {weight_file} and {sv_file}...")
        
    weight_data = {
        "weights": weights.tolist() if weights is not None else [],
        "bias": float(bias) if bias is not None else 0.0
    }
    
    sv_data = {
        "seperable": int(is_separable),
        "support_vectors": support_vectors if is_separable else []
    }
    
    with open(weight_file, 'w') as f:
        json.dump(weight_data, f, indent=2)
    print(f"Weights and bias saved to {weight_file}.")
    
    with open(sv_file, 'w') as f:
        json.dump(sv_data, f, indent=2)
    print(f"Support vector information saved to {sv_file}.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python svm.py train_filename.csv")
        return
    
    train_file = sys.argv[1]
    base_name = train_file.split('train_')[1].split('.')[0]
    
    weight_file = f"weight_{base_name}.json"
    sv_file = f"sv_{base_name}.json"
    
    X, y = load_data(train_file)
    weights, bias, is_separable = train_svm(X, y)
    
    support_vectors = []
    if is_separable and weights is not None and bias is not None:
        support_vectors = find_support_vectors(X, y, weights, bias)
        
    save_results(weights, bias, is_separable, support_vectors, weight_file, sv_file)
    print("Processing complete.")

if __name__ == "__main__":
    main()
