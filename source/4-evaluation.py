import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import seaborn as sns
import pytorch_lightning as pl
from tqdm import tqdm
from models import subj, classn
from models.lightning import BaseModel, DataModule
from torchmetrics import Accuracy, F1Score, Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix


cross_evaluation = False  # if True, will classify the dataset 2 with model 1 (subjectivity), else, will evaluate the selected dataset

with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    TASK = settings["task"]
    NN_MODEL = settings["subj_model"]
    CHOSEN_MODEL = settings["subj_chosen_model"]
    DATASET = settings["dataset_to_evaluate"]
    DATASET_STAGE = settings["dataset_stage"]
    TRAIN_WITH_WEIGHTS = "train_with_weights" if settings["train_subj_with_weights"] else "no_weight"

DATASET_NAME = {1: "subjectivity",
                2: "climate change"}
DATASET_NAME = DATASET_NAME[DATASET]

STAGES_MAPPING = {"fit": "train",
                  "validate": "val",
                  "test": "test"}

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{TASK}", f"{NN_MODEL}")
CHECKPOINT_MODEL = os.path.join(CHECKPOINT_PATH, f"{CHOSEN_MODEL}.ckpt")

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "embeddings")

networks_choices = {"subj_classifier": {"subj_classifier_v1": subj.subj_classifier_v1,
                                        "subj_classifier_v2": subj.subj_classifier_v2,
                                        "subj_classifier_v3": subj.subj_classifier_v3},
                      "cc_classifier": {}}
MODEL_NETWORK = networks_choices[TASK][NN_MODEL]

#loss_choices = {"binary_cross_entropy":torch.nn.BCEWithLogitsLoss()}
def main():
    if not cross_evaluation:
        RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        datamodule = DataModule(datasets_root=DATASET_PATH, dataset_id=DATASET, batch_size=1)
        datamodule.setup(DATASET_STAGE)

        if DATASET_STAGE == "test":
            inference_dataset = datamodule.test_dataloader()
        elif DATASET_STAGE == "validate":
            inference_dataset = datamodule.val_dataloader()
        elif DATASET_STAGE == "fit":
            inference_dataset = datamodule.train_dataloader()
        else: raise Exception(f"Invalid option ({DATASET_STAGE})\nExpected: fit (train), validate (val), test (test)")

        model = BaseModel.load_from_checkpoint(CHECKPOINT_MODEL,
                                                model=MODEL_NETWORK,
                                                loss_function=torch.nn.BCEWithLogitsLoss(),
                                                strict=False).eval()
        sigmoid = torch.nn.Sigmoid()
        predictions = []
        real_values = []
        with torch.no_grad():
            for x, y in tqdm(inference_dataset, desc="calculating inferences"):
                output = model(x)
                probs = sigmoid(output)
                predictions.append(probs)
                real_values.append(y)
                predictions.append(probs.cpu())
                real_values.append(y.cpu())

        predictions = torch.cat(predictions, dim=0).squeeze()
        real_values = torch.cat(real_values, dim=0).squeeze()
        pred_labels = (predictions >= 0.5).long() # the predicted class
        true_labels = real_values.long()

        accuracy = Accuracy(task="binary")
        precision = Precision(task="binary")
        recall = Recall(task="binary")
        f1 = F1Score(task="binary")

        acc = accuracy(pred_labels, true_labels)
        prec = precision(pred_labels, true_labels)
        rec = recall(pred_labels, true_labels)
        f1_score = f1(pred_labels, true_labels)

        cm = confusion_matrix(true_labels.numpy(), pred_labels.numpy())
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Class OBJ', 'Class SUBJ'],
                   yticklabels=['Class OBJ', 'Class SUBJ'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        metrics_text = f'Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1-Score:  {f1_score:.4f}'
        plt.text(1.25, 0.5, metrics_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.title(f'Confusion Matrix - Dataset {DATASET_NAME} (stage: {STAGES_MAPPING[DATASET_STAGE]})')
        plt.tight_layout()

        plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{DATASET_NAME}_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_MODEL})_{TRAIN_WITH_WEIGHTS}.png'), dpi=300)

        # Save metrics to file
        metrics_dict = {
            'dataset': DATASET,
            'model': CHOSEN_MODEL,
            'accuracy': acc.item(),
            'precision': prec.item(),
            'recall': rec.item(),
            'f1_score': f1_score.item(),
            'total_samples': len(true_labels)
        }

        metrics_df = pd.DataFrame([metrics_dict])
        metrics_file = os.path.join(RESULTS_DIR, f'metrics({DATASET_NAME})_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_MODEL})_{TRAIN_WITH_WEIGHTS}.csv')
        metrics_df.to_csv(metrics_file, index=False)


    elif (cross_evaluation) and (DATASET==2):
        pass
    elif (cross_evaluation) and (DATASET==1):
        raise Exception(f"cross evaluation only available for climate change dataset (2)\nChosen dataset: {DATASET}")


if __name__ == "__main__":
    main()
