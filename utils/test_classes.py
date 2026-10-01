import os
import glob

def extract_labels(dataset_path):
    labels = set()
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.wav'):
                # Assuming format: idx_label.jpg or idx_label.wav
                parts = file.split('_', 1)  # Split only on first underscore
                if len(parts) == 2:
                    label_with_ext = parts[1]
                    label = label_with_ext.rsplit('.', 1)[0]  # Remove extension
                    labels.add(label)
    return sorted(labels)

if __name__ == "__main__":
    dataset_path = '/home/singh_prajwal/dataset/IS3'
    labels = extract_labels(dataset_path)
    print("All unique labels:")
    for label in labels:
        print(label)
    
    # Optionally save to a file
    with open('all_labels.txt', 'w') as f:
        for label in labels:
            f.write(label + '\n')
    print(f"\nLabels saved to all_labels.txt")