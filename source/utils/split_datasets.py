"""This function splits the dataset into smaller size for better processing (classification) for the auto-encoder models"""

import pandas as pd
import os
from tqdm import tqdm

datasets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "processed_datasets")
datasets = {name : pd.read_csv(datasets_dir, f"{name}.csv") for name in ["train_dataset_2", "val_dataset_2", "test_dataset_2"]}

chunk_size = 2000
output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "processed_datasets", "chunks")
os.makedirs(output_dir, exist_ok=True)

for dataset_name, df in datasets.items():
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Arredonda para cima

    for i in tqdm(range(num_chunks), desc="processing"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)

        chunk_df = df.iloc[start_idx:end_idx]

        chunk_filename = f"{dataset_name}_part{i+1}.txt"
        chunk_path = os.path.join(output_dir, chunk_filename)

        # Saves the splites dataset
        chunk_df.to_csv(chunk_path, index=False)
