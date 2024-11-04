import pandas as pd

# Load the predictions file (2000 predictions, 0 or 1, no header)
predictions = pd.read_csv('predictions_real_pruned_train.csv', header=None, names=['Prediction'])

# Load the test_real.csv file and select the target column
test_data = pd.read_csv('train_real.csv')
target = test_data['target']  # Replace 'target' with the actual name of the target column if different

# Check if both files have the same length
if len(predictions) != len(target):
    raise ValueError("The number of predictions does not match the number of target labels.")

# Calculate accuracy
accuracy = (predictions['Prediction'] == target).mean()

print(f"Accuracy: {accuracy * 100:.2f}%")