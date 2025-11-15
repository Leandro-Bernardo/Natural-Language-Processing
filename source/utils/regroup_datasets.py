"""This function regroupts the splited datasets into original size after processing (classification) for the auto-encoder model"""

import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict

datasets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "processed_datasets", "GPTed_chunks")
output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "processed_datasets")
os.makedirs(output_dir, exist_ok=True)

# Lista todos os arquivos CSV no diretório
all_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]

grouped_files = defaultdict(list)
for filename in all_files:
    base_name = filename.replace('_classified.csv', '').replace('_labeled.csv', '')
    parts = base_name.split('_part')
    if len(parts) == 2:
        dataset_base = parts[0]
        part_number = int(parts[1])
        grouped_files[dataset_base].append((part_number, filename))

for dataset_name, files in grouped_files.items():
    files.sort(key=lambda x: x[0])
    dfs = []

    for part_num, filename in tqdm(files, desc=f"Reading {dataset_name}"):
        file_path = os.path.join(datasets_dir, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    output_filename = f"{dataset_name}_classified.csv"
    output_path = os.path.join(output_dir, output_filename)
    combined_df.to_csv(output_path, index=False)

