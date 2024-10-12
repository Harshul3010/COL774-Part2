import numpy as np
import sys

# Define the class mapping as used in the model
class_mapping = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']

def validate_predictions(npy_path, output_path):
    # Load the .npy file with class probabilities
    prob_arrays = np.load(npy_path)

    # Determine inferred predictions from the .npy file
    inferred_predictions = [class_mapping[np.argmax(prob_array)] for prob_array in prob_arrays]

    # Load your predictions from output.txt
    with open(output_path, 'r', encoding='utf-8') as f:
        my_predictions = [line.strip() for line in f]

    # Check that both lists are the same length
    if len(my_predictions) != len(inferred_predictions):
        raise ValueError("The number of predictions in output.txt does not match the number of inferred predictions.")

    # Calculate accuracy by comparing the two lists
    correct = sum(1 for my_pred, inferred_pred in zip(my_predictions, inferred_predictions) if my_pred == inferred_pred)
    accuracy = correct / len(inferred_predictions)

    # Print results
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {correct} out of {len(inferred_predictions)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python chek.py path_to_npy path_to_output")
        sys.exit(1)

    npy_path = sys.argv[1]
    output_path = sys.argv[2]
    validate_predictions(npy_path, output_path)
