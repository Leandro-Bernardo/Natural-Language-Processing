#from transformers import AutoTokenizer, AutoModel
import torch
import os
import pandas as pd
import json, yaml
from sentence_transformers import SentenceTransformer
from torch import FloatTensor, UntypedStorage
from tqdm import tqdm
from typing import Optional


with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    TASK = settings["task"]

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DATASETS_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed_datasets")
EMBEDDINGS_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "embeddings")
os.makedirs(EMBEDDINGS_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(EMBEDDINGS_SAVE_PATH, "metadata"), exist_ok=True)


# loads the tokenizer and the model
model = SentenceTransformer(MODEL)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

# loads the data
if TASK == "subj_classifier":
    train_dataset = pd.read_csv(os.path.join(DATASETS_PATH, f"train_dataset_1.csv")).rename({"1":"sentence", "2":"label"}, axis=1).dropna()
    val_dataset = pd.read_csv(os.path.join(DATASETS_PATH, f"val_dataset_1.csv")).rename({"1":"sentence", "2":"label"}, axis=1).dropna()
    test_dataset = pd.read_csv(os.path.join(DATASETS_PATH, f"test_dataset_1.csv")).rename({"0":"sentence", "1":"label"}, axis=1).dropna()
    label_mapper = None
    dataset_id = 1
elif TASK == "multitask_classifier":
    assert any("classified.csv" in x.split("_") for x in os.listdir(DATASETS_PATH)), "Climate change dataset must get pseudo labels of subjectivity. Use a pre-trained model or a decoder-only model to do so. The file format must be *_dataset_classified.csv"
    train_dataset = pd.read_csv(os.path.join(DATASETS_PATH, "train_dataset_classified.csv")).rename({"0":"sentence", "1":"label"}, axis=1).dropna()
    val_dataset = pd.read_csv(os.path.join(DATASETS_PATH, "val_dataset_classified.csv")).rename({"0":"sentence", "1":"label"}, axis=1).dropna()
    test_dataset = pd.read_csv(os.path.join(DATASETS_PATH, "test_dataset_classified.csv")).rename({"0":"sentence", "1":"label"}, axis=1).dropna()
    unique_train_labels = train_dataset['label'].unique()
    unique_val_labels = val_dataset['label'].unique()
    unique_test_labels = test_dataset['label'].unique()
    assert (len(unique_val_labels) == len(unique_train_labels)) & (len(unique_test_labels) == len(unique_train_labels)) & (len(unique_test_labels) == len(unique_val_labels)), "some class(es) are not present in all train/val/test dataset"
    label_mapper = {label: idx for idx, label in enumerate(unique_train_labels)}
    with open(os.path.join(os.path.join(EMBEDDINGS_SAVE_PATH, "metadata"), f"cc_classes.json"), "w") as file:
        json.dump({
            "total_classes": len(label_mapper.keys()),
            "mapper": label_mapper,
        }, file)
    dataset_id = 2

def extract_embeddings(train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, test_dataset: pd.DataFrame, dataset_id: int, task:str, mapper: Optional[pd.DataFrame]):
    # prepare the sentences to extract embeddings
    train_sentences = train_dataset['sentence'].tolist()
    val_sentences = val_dataset['sentence'].tolist()
    test_sentences = test_dataset['sentence'].tolist()
        # extract embeddings
    train_embeddings = model.encode(train_sentences, show_progress_bar=True)
    val_embeddings = model.encode(val_sentences, show_progress_bar=True)
    test_embeddings = model.encode(test_sentences, show_progress_bar=True)

    # converts embeddings to torch.tensor
    train_embeddings = torch.from_numpy(train_embeddings)
    val_embeddings = torch.from_numpy(val_embeddings)
    test_embeddings = torch.from_numpy(test_embeddings)

    # gets shapes for save embeddings raw file
    train_size = train_embeddings.shape
    val_size = val_embeddings.shape
    test_size = test_embeddings.shape

    # creat untyped storage object for save the embeddings
    train_dataset_storage = FloatTensor(torch.from_file(
                                                        os.path.join(EMBEDDINGS_SAVE_PATH, f"embeddings_train_{dataset_id}.bin"),
                                                        shared = True, size= (train_size[0]*train_size[1]))).view(train_size[0], train_size[1])
    val_dataset_storage = FloatTensor(torch.from_file(
                                                        os.path.join(EMBEDDINGS_SAVE_PATH, f"embeddings_val_{dataset_id}.bin"),
                                                        shared = True, size= (val_size[0]*val_size[1]))).view(val_size[0], val_size[1])
    test_dataset_storage = FloatTensor(torch.from_file(
                                                        os.path.join(EMBEDDINGS_SAVE_PATH, f"embeddings_test_{dataset_id}.bin"),
                                                        shared = True, size= (test_size[0]*test_size[1]))).view(test_size[0], test_size[1])
    # prepare the single task labels
    if task == "subj_classifier":
        train_labels = train_dataset['label'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
        val_labels = val_dataset['label'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
        test_labels = test_dataset['label'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
        # converts data to tensor
        train_labels = torch.from_numpy(train_labels).unsqueeze(dim=1)
        val_labels = torch.from_numpy(val_labels).unsqueeze(dim=1)
        test_labels = torch.from_numpy(test_labels).unsqueeze(dim=1)
        # gets shapes for save raw file
        train_labels_shape = train_labels.shape
        val_labels_shape = val_labels.shape
        test_labels_shape = test_labels.shape
        # creat untyped storage object for save the labels (single task)
        train_labels_storage = FloatTensor(torch.from_file(
                                                            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_train_{dataset_id}.bin"),
                                                            shared=True, size=(train_labels_shape[0]*train_labels_shape[1]))).view(train_labels_shape[0], train_labels_shape[1])
        val_labels_storage = FloatTensor(torch.from_file(
                                                            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_val_{dataset_id}.bin"),
                                                            shared=True, size=(val_labels_shape[0]*val_labels_shape[1]))).view(val_labels_shape[0], val_labels_shape[1])
        test_labels_storage = FloatTensor(torch.from_file(
                                                            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_test_{dataset_id}.bin"),
                                                            shared=True, size=(test_labels_shape[0]*test_labels_shape[1]))).view(test_labels_shape[0], test_labels_shape[1])
        # writes the embeddings and the labels on the created untyped storage object
        for i in tqdm(range(0, len(train_embeddings)), desc="saving train data"):
            train_dataset_storage[i,:] = train_embeddings[i,:]
            train_labels_storage[i,:] = train_labels[i,:]
        for i in tqdm(range(0, len(val_dataset)), desc="saving val data"):
            val_dataset_storage[i,:] = val_embeddings[i,:]
            val_labels_storage[i,:] = val_labels[i,:]
        for i in tqdm(range(0, len(test_dataset)), desc="saving test data"):
            test_dataset_storage[i,:] = test_embeddings[i,:]
            test_labels_storage[i,:] = test_labels[i,:]

    # prepare the multitask labels
    elif task == "multitask_classifier":
        train_labels = train_dataset.loc[:,['subjectivity', "label"]]
        val_labels = val_dataset.loc[:,['subjectivity', "label"]]
        test_labels = test_dataset.loc[:,['subjectivity', "label"]]
        # maps subj labels
        train_subj_labels = train_labels['subjectivity'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
        val_subj_labels = val_labels['subjectivity'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
        test_subj_labels = test_labels['subjectivity'].apply(lambda x: 1 if x == 'SUBJ' else 0).values
        # maps cc labels
        train_cc_labels = train_labels['label'].map(mapper).values
        val_cc_labels = val_labels['label'].map(mapper).values
        test_cc_labels = test_labels['label'].map(mapper).values
        # converts data to tensor
        train_subj_labels = torch.from_numpy(train_subj_labels).float()
        val_subj_labels = torch.from_numpy(val_subj_labels).float()
        test_subj_labels = torch.from_numpy(test_subj_labels).float()
        train_cc_labels = torch.from_numpy(train_cc_labels).long()
        val_cc_labels = torch.from_numpy(val_cc_labels).long()
        test_cc_labels = torch.from_numpy(test_cc_labels).long()
        # creat untyped storage object for save the labels (multitask)
            # Subjectivity labels storage
        train_subj_storage = FloatTensor(torch.from_file(
            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_subj_train_{dataset_id}.bin"),
            shared=True, size=len(train_subj_labels)))
        val_subj_storage = FloatTensor(torch.from_file(
            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_subj_val_{dataset_id}.bin"),
            shared=True, size=len(val_subj_labels)))
        test_subj_storage = FloatTensor(torch.from_file(
            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_subj_test_{dataset_id}.bin"),
            shared=True, size=len(test_subj_labels)))

            # Climate Change labels storage
        train_cc_storage = FloatTensor(torch.from_file(
            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_cc_train_{dataset_id}.bin"),
            shared=True, size=len(train_cc_labels)))
        val_cc_storage = FloatTensor(torch.from_file(
            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_cc_val_{dataset_id}.bin"),
            shared=True, size=len(val_cc_labels)))
        test_cc_storage = FloatTensor(torch.from_file(
            os.path.join(EMBEDDINGS_SAVE_PATH, f"labels_cc_test_{dataset_id}.bin"),
            shared=True, size=len(test_cc_labels)))
        #writes the embeddings and the labels on the created untyped storage object
        for i in tqdm(range(0, len(train_embeddings)), desc="saving train data"):
            train_dataset_storage[i,:] = train_embeddings[i,:]
            train_subj_storage[i] = train_subj_labels[i]
            train_cc_storage[i] = train_cc_labels[i]
        for i in tqdm(range(0, len(val_dataset)), desc="saving val data"):
            val_dataset_storage[i,:] = val_embeddings[i,:]
            val_subj_storage[i] = val_subj_labels[i]
            val_cc_storage[i] = val_cc_labels[i]
        for i in tqdm(range(0, len(test_dataset)), desc="saving test data"):
            test_dataset_storage[i,:] = test_embeddings[i,:]
            test_subj_storage[i] = test_subj_labels[i]
            test_cc_storage[i] = test_cc_labels[i]

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
    print(f"\nExtracting embeddings for {TASK}\n")
    extract_embeddings(train_dataset, val_dataset, test_dataset, dataset_id, TASK, label_mapper)



