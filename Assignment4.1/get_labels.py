import pandas as pd

def save_test_labels(test_file, output_file):
    
    # Read the test.tsv file
    df_test = pd.read_csv(test_file, sep="\t", header=None, quoting=3, encoding='utf-8')
    
    # Extract labels from the second column (index 1)
    labels = df_test[1].tolist()
    
    # Write labels to output_test.txt, one label per line
    with open(output_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label}\n")

save_test_labels('valid.tsv', 'output_test.txt')