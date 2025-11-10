#from transformers import AutoTokenizer, AutoModel
import torch
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch import FloatTensor, UntypedStorage
from tqdm import tqdm
import json


MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DATASETS_PATH = os.path.join(".", "dataset", "processed_datasets")
EMBEDDINGS_SAVE_PATH = os.path.join(".", "dataset", "embeddings")
os.makedirs(EMBEDDINGS_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(EMBEDDINGS_SAVE_PATH, "metadata"), exist_ok=True)

# loads the tokenizer and the model
model = SentenceTransformer(MODEL)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

# loads the data
train_dataset_1 = pd.read_csv(os.path.join(DATASETS_PATH, f"train_dataset_1.csv")).rename({"1":"sentence", "2":"label"}, axis=1)
val_dataset_1 = pd.read_csv(os.path.join(DATASETS_PATH, f"val_dataset_1.csv")).rename({"1":"sentence", "2":"label"}, axis=1)
test_dataset_1 = pd.read_csv(os.path.join(DATASETS_PATH, f"test_dataset_1.csv")).rename({"0":"sentence", "1":"label"}, axis=1)
# TODO not used yet
train_dataset_2 = pd.read_csv(os.path.join(DATASETS_PATH, "train_dataset_2.csv")).rename({"0":"sentence", "1":"label"}, axis=1)
val_dataset_2 = pd.read_csv(os.path.join(DATASETS_PATH, "val_dataset_2.csv")).rename({"0":"sentence", "1":"label"}, axis=1)
test_dataset_2 = pd.read_csv(os.path.join(DATASETS_PATH, "test_dataset_2.csv")).rename({"0":"sentence", "1":"label"}, axis=1)


def extract_embeddings(train_dataset, val_dataset, test_dataset, dataset_id):
    # prepare the sentences to extract embeddings
    train_sentences = train_dataset['sentence'].tolist()
    val_sentences = val_dataset['sentence'].tolist()
    test_sentences = test_dataset['sentence'].tolist()
    # prepare the labels
    train_labels = train_dataset['label'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
    val_labels = val_dataset['label'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
    test_labels = test_dataset['label'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
    # extract embeddings
    train_embeddings = model.encode(train_sentences, show_progress_bar=True)
    val_embeddings = model.encode(val_sentences, show_progress_bar=True)
    test_embeddings = model.encode(test_sentences, show_progress_bar=True)

    # converts data to torch.tensor
    train_embeddings = torch.from_numpy(train_embeddings)
    val_embeddings = torch.from_numpy(val_embeddings)
    test_embeddings = torch.from_numpy(test_embeddings)

    train_labels = torch.from_numpy(train_labels).unsqueeze(dim=1)
    val_labels = torch.from_numpy(val_labels).unsqueeze(dim=1)
    test_labels = torch.from_numpy(test_labels).unsqueeze(dim=1)

    # gets shapes for save raw file
    train_size = train_embeddings.shape
    val_size = val_embeddings.shape
    test_size = test_embeddings.shape

    train_labels_shape = train_labels.shape
    val_labels_shape = val_labels.shape
    test_labels_shape = test_labels.shape

    # creat untyped storage object for save the descriptors
    train_dataset_storage = FloatTensor(torch.from_file(os.path.join(EMBEDDINGS_SAVE_PATH, f"embeddings_train_{dataset_id}.bin"), shared = True, size= (train_size[0]*train_size[1]))).view(train_size[0], train_size[1])
    train_labels_storage = FloatTensor(torch.from_file(os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_train_{dataset_id}.bin"), shared = True, size= (train_labels_shape[0]*train_labels_shape[1]))).view(train_labels_shape[0], train_labels_shape[1])
    val_dataset_storage = FloatTensor(torch.from_file(os.path.join(EMBEDDINGS_SAVE_PATH, f"embeddings_val_{dataset_id}.bin"), shared = True, size= (val_size[0]*val_size[1]))).view(val_size[0], val_size[1])
    val_labels_storage = FloatTensor(torch.from_file(os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_val_{dataset_id}.bin"), shared = True, size= (val_labels_shape[0]*val_labels_shape[1]))).view(val_labels_shape[0], val_labels_shape[1])
    test_dataset_storage = FloatTensor(torch.from_file(os.path.join(EMBEDDINGS_SAVE_PATH, f"embeddings_test_{dataset_id}.bin"), shared = True, size= (test_size[0]*test_size[1]))).view(test_size[0], test_size[1])
    test_labels_storage = FloatTensor(torch.from_file(os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_test_{dataset_id}.bin"), shared = True, size= (test_labels_shape[0]*test_labels_shape[1]))).view(test_labels_shape[0], test_labels_shape[1])

    # writes the embeddings on the created untyped storage object
    for i in tqdm(range(0, len(train_embeddings)), desc="saving train embeddings"):
        train_dataset_storage[i,:] = train_embeddings[i,:]
        train_labels_storage[i,:] = train_labels[i,:]
    for i in tqdm(range(0, len(val_dataset)), desc="saving val embeddings"):
        val_dataset_storage[i,:] = val_embeddings[i,:]
        val_labels_storage[i,:] = val_labels[i,:]
    for i in tqdm(range(0, len(test_dataset)), desc="saving test embeddings"):
        test_dataset_storage[i,:] = test_embeddings[i,:]
        test_labels_storage[i,:] = test_labels[i,:]

    # writes metadata
    with open(os.path.join(os.path.join(EMBEDDINGS_SAVE_PATH, "metadata"), f"metadata_train_{dataset_id}.json"), "w") as file:
        json.dump({
            "total_sentences": train_size[0],
            "embedding_size": train_size[1],
        }, file)
    with open(os.path.join(os.path.join(EMBEDDINGS_SAVE_PATH, "metadata"), f"metadata_val_{dataset_id}.json"), "w") as file:
        json.dump({
            "total_sentences": val_size[0],
            "embedding_size": val_size[1],
        }, file)
    with open(os.path.join(os.path.join(EMBEDDINGS_SAVE_PATH, "metadata"), f"metadata_test_{dataset_id}.json"), "w") as file:
        json.dump({
            "total_sentences": test_size[0],
            "embedding_size": test_size[1],
        }, file)




if __name__ == "__main__":
    extract_embeddings(train_dataset_1, val_dataset_1, test_dataset_1, 1)
    extract_embeddings(train_dataset_2, val_dataset_2, test_dataset_2, 2)



