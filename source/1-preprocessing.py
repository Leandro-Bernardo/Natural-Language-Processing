import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_1_PATH = os.path.join(".", "dataset", "subjectivity")
DATASET_2_PATH = os.path.join(".", "dataset", "classification")

dataset_1 = {
    os.path.splitext(filename)[0]: pd.read_csv(
        os.path.join(DATASET_1_PATH, filename),
        sep="\t",
        header=None
    )
    for filename in os.listdir(DATASET_1_PATH)
    if filename.endswith(".tsv")
}
dataset_2 = {
    os.path.splitext(filename)[0]: pd.read_csv(
        os.path.join(DATASET_2_PATH, filename),
        sep="\t",
        header=None
    )
    for filename in os.listdir(DATASET_2_PATH)
    if filename.endswith(".txt")
}
#dataset_1 = {filename.strip('.tsv'): pd.read_csv(os.path.join(DATASET_1_PATH, filename), sep='\t', header=None) for filename in os.listdir(DATASET_1_PATH)}
#dataset_2 = {filename.strip('.txt'): pd.read_csv(os.path.join(DATASET_2_PATH, filename), sep='\t', header=None) for filename in os.listdir(DATASET_2_PATH)}

# splits datasets into train, val, test (train and val with holdout 0.8/0.2)
train_dataset_1 = dataset_1["train_en"].loc[1:,1:2]
train_dataset_1, val_dataset_1 = train_test_split(train_dataset_1, test_size=0.2, random_state=42, shuffle=True, stratify=train_dataset_1.loc[:, 2])
test_dataset_1 = dataset_1["test_en_labeled"].loc[1:,0:1]

train_dataset_2 = dataset_2["4C_training_CLEANED"]
train_dataset_2, val_dataset_2 = train_test_split(train_dataset_2, test_size=0.2, random_state=42, shuffle=True, stratify=train_dataset_2.loc[:, 1])
test_dataset_2 = dataset_2["4C_test_CLEANED"]

# saves datasets
os.makedirs('dataset/processed_datasets', exist_ok=True)
train_dataset_1.to_csv('dataset/processed_datasets/train_dataset_1.csv', index=False)
val_dataset_1.to_csv('dataset/processed_datasets/val_dataset_1.csv', index=False)
test_dataset_1.to_csv('dataset/processed_datasets/test_dataset_1.csv', index=False)

train_dataset_2.to_csv('dataset/processed_datasets/train_dataset_2.csv', index=False)
val_dataset_2.to_csv('dataset/processed_datasets/val_dataset_2.csv', index=False)
test_dataset_2.to_csv('dataset/processed_datasets/test_dataset_2.csv', index=False)


print(" ")