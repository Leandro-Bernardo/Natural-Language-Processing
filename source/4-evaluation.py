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
from models.lightning import BaseModel, MultitaskModel,  DataModule
from torchmetrics import Accuracy, F1Score, Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix

with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    TASK = settings["task"]
    NN_SUBJ_MODEL = settings["subj_model"]
    NN_MULTITASK_MODEL = settings["multitask_model"]
    CHOSEN_MULTITASK_MODEL = settings["multitask_chosen_model"]
    CHOSEN_SUBJ_MODEL = settings["subj_chosen_model"] if TASK == "subj_classifier" else f"{settings['multitask_chosen_model']}_subj_updated"
    DATASET = settings["dataset_to_evaluate"]
    DATASET_STAGE = settings["dataset_stage"]
    TRAIN_WITH_WEIGHTS = "train_with_weights" if settings["train_subj_with_weights"] else "no_weight"

DATASET_NAME = {1: "subjectivity",
                2: "climate change"}
DATASET_NAME = DATASET_NAME[DATASET]

STAGES_MAPPING = {"fit": "train",
                  "validate": "val",
                  "test": "test"}

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "embeddings")

networks_choices = {"subj_classifier": {"subj_classifier_v1": subj.subj_classifier_v1,
                                        "subj_classifier_v2": subj.subj_classifier_v2,
                                        "subj_classifier_v3": subj.subj_classifier_v3},
                    "multitask_classifier": {"multitask_model_v3": classn.multitask_model_v3}}
if TASK == "subj_classifier":
    MODEL_NETWORK = networks_choices[TASK][NN_SUBJ_MODEL]

    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{TASK}", f"{NN_SUBJ_MODEL}")
    CHECKPOINT_MODEL = os.path.join(CHECKPOINT_PATH, f"{CHOSEN_SUBJ_MODEL}.ckpt")

elif TASK == "multitask_classifier":
    MODEL_SUBJ_NETWORK = networks_choices["subj_classifier"][NN_SUBJ_MODEL]
    MODEL_CC_NETWORK = networks_choices[TASK][NN_MULTITASK_MODEL]

    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{TASK}", f"{NN_MULTITASK_MODEL}")
    CHECKPOINT_SUBJ_MODEL = os.path.join(CHECKPOINT_PATH, f"{CHOSEN_SUBJ_MODEL}.ckpt")
    CHECKPOINT_MULTITASK_MODEL = os.path.join(CHECKPOINT_PATH, f"{CHOSEN_MULTITASK_MODEL}.ckpt")


def main():
    if TASK == "subj_classifier":
        RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "subj")
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
                predictions.append(probs.cpu())
                real_values.append(y.cpu())

        predictions = torch.cat(predictions, dim=0).squeeze()
        real_values = torch.cat(real_values, dim=0).squeeze()
        pred_labels = (predictions >= 0.5).long()
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

        plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{DATASET_NAME}_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_SUBJ_MODEL})_{TRAIN_WITH_WEIGHTS}.png'), dpi=300)

        metrics_dict = {
            'dataset': DATASET,
            'model': CHOSEN_SUBJ_MODEL,
            'accuracy': acc.item(),
            'precision': prec.item(),
            'recall': rec.item(),
            'f1_score': f1_score.item(),
            'total_samples': len(true_labels)
        }

        metrics_df = pd.DataFrame([metrics_dict])
        metrics_file = os.path.join(RESULTS_DIR, f'metrics({DATASET_NAME})_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_SUBJ_MODEL})_{TRAIN_WITH_WEIGHTS}.csv')
        metrics_df.to_csv(metrics_file, index=False)

    elif TASK == "multitask_classifier":
        NUM_CLASSES = 18
        RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "multitask")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        datamodule = DataModule(datasets_root=DATASET_PATH, dataset_id=DATASET, batch_size=1)
        datamodule.setup(DATASET_STAGE)

        if DATASET_STAGE == "test":
            inference_dataset = datamodule.test_dataloader()
        elif DATASET_STAGE == "validate":
            inference_dataset = datamodule.val_dataloader()
        elif DATASET_STAGE == "fit":
            inference_dataset = datamodule.train_dataloader()
        else:
            raise Exception(f"Invalid option ({DATASET_STAGE})\nExpected: fit (train), validate (val), test (test)")

        model_subj = BaseModel.load_from_checkpoint(CHECKPOINT_SUBJ_MODEL,
                                                model = MODEL_SUBJ_NETWORK,
                                                loss_function = torch.nn.BCEWithLogitsLoss(),
                                                strict = False).eval()
        model_multitask = MultitaskModel.load_from_checkpoint(CHECKPOINT_MULTITASK_MODEL,
                                                cc_model = lambda: MODEL_CC_NETWORK(num_classes=18),
                                                subj_trained_model = model_subj,
                                                subj_loss = torch.nn.BCEWithLogitsLoss(),
                                                cc_loss = torch.nn.CrossEntropyLoss(),
                                                num_cc_classes = 18,
                                                strict = False).eval()
        model_cc = model_multitask.cc_model

        subj_predictions = []
        subj_real_values = []
        cc_predictions = []
        cc_real_values = []
        with torch.no_grad():
            for X, (y1, y2) in tqdm(inference_dataset, desc="calculating inferences"):
                subj_output = model_subj(X)
                subj_predictions.append(subj_output.cpu())
                subj_real_values.append(y1.cpu())

                cc_output = model_cc(X)
                cc_predictions.append(cc_output.cpu())
                cc_real_values.append(y2.cpu())

        subj_predictions = torch.cat(subj_predictions, dim=0)
        subj_real_values = torch.cat(subj_real_values, dim=0).squeeze()
        cc_output = torch.cat(cc_predictions, dim=0)
        cc_real_values = torch.cat(cc_real_values, dim=0).squeeze()

        subj_pred_labels = (subj_predictions >= 0.5).squeeze().long()
        subj_true_labels = subj_real_values.long()
        cc_pred_labels = torch.tensor([torch.argmax(prediction, dim=1) for prediction in cc_predictions])
        cc_true_labels = cc_real_values.long()

        subj_accuracy = Accuracy(task="binary")
        subj_precision = Precision(task="binary")
        subj_recall = Recall(task="binary")
        subj_f1 = F1Score(task="binary")
        cc_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        cc_precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro')
        cc_recall = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro')
        cc_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average='macro')

        subj_acc = subj_accuracy(subj_pred_labels, subj_true_labels)
        subj_prec = subj_precision(subj_pred_labels, subj_true_labels)
        subj_rec = subj_recall(subj_pred_labels, subj_true_labels)
        subj_f1_score = subj_f1(subj_pred_labels, subj_true_labels)

        cc_acc = cc_accuracy(cc_pred_labels, cc_true_labels)
        cc_prec = cc_precision(cc_pred_labels, cc_true_labels)
        cc_rec = cc_recall(cc_pred_labels, cc_true_labels)
        cc_f1_score = cc_f1(cc_pred_labels, cc_true_labels)

        # print(f"\n{'='*60}")
        # print(f"Overall Metrics - {DATASET_NAME} ({STAGES_MAPPING[DATASET_STAGE]})")
        # print(f"{'='*60}")
        # print(f"Accuracy:  {acc:.4f}")
        # print(f"Precision: {prec:.4f} (macro)")
        # print(f"Recall:    {rec:.4f} (macro)")
        # print(f"F1-Score:  {f1_score:.4f} (macro)")
        # print(f"{'='*60}\n")

        # print("\nDetailed Classification Report:")
        # print(classification_report(true_labels.numpy(), pred_labels.numpy(),
        #                            target_names=[f'Class {i}' for i in range(NUM_CLASSES)],
        #                            digits=4))

        subj_cm = confusion_matrix(subj_true_labels.numpy(), subj_pred_labels.numpy())
        cc_cm = confusion_matrix(cc_true_labels.numpy(), cc_pred_labels.numpy())

        #subj matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(subj_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Class OBJ', 'Class SUBJ'],
                   yticklabels=['Class OBJ', 'Class SUBJ'],
                   cbar_kws={'label': 'Count'})
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        metrics_text = f'Accuracy:  {subj_acc:.4f}\nPrecision: {subj_prec:.4f}\nRecall:    {subj_rec:.4f}\nF1-Score:  {subj_f1_score:.4f}'
        plt.text(1.02, 0.5, metrics_text,
                transform=plt.gca().transAxes,
                fontsize=11,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.title(f'Confusion Matrix - Dataset {DATASET_NAME} (subj) (stage: {STAGES_MAPPING[DATASET_STAGE]})\nMulticlass Classification (2 classes)',
                 fontsize=14, pad=20)
        plt.tight_layout()

        plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{DATASET_NAME}(subj)_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_MULTITASK_MODEL})_{TRAIN_WITH_WEIGHTS}_multiclass.png'), dpi=300)

        # cc matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(cc_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'C{i}' for i in range(NUM_CLASSES)],
                   yticklabels=[f'C{i}' for i in range(NUM_CLASSES)],
                   cbar_kws={'label': 'Count'})
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        metrics_text = f'Accuracy:  {cc_acc:.4f}\nPrecision: {cc_prec:.4f}\nRecall:    {cc_rec:.4f}\nF1-Score:  {cc_f1_score:.4f}'
        plt.text(1.02, 0.5, metrics_text,
                transform=plt.gca().transAxes,
                fontsize=11,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.title(f'Confusion Matrix - Dataset {DATASET_NAME}(cc) (stage: {STAGES_MAPPING[DATASET_STAGE]})\nMulticlass Classification ({NUM_CLASSES} classes)',
                 fontsize=14, pad=20)
        plt.tight_layout()

        plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{DATASET_NAME}(cc)_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_MULTITASK_MODEL})_{TRAIN_WITH_WEIGHTS}_multiclass.png'), dpi=300)

        # subj metrics
        metrics_dict = {
            'dataset': DATASET,
            'model': CHOSEN_SUBJ_MODEL,
            'accuracy': subj_acc.item(),
            'precision': subj_prec.item(),
            'recall': subj_rec.item(),
            'f1_score': subj_f1_score.item(),
            'total_samples': len(subj_true_labels)
        }

        metrics_df = pd.DataFrame([metrics_dict])
        metrics_file = os.path.join(RESULTS_DIR, f'metrics({DATASET_NAME})(subj)_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_SUBJ_MODEL})_{TRAIN_WITH_WEIGHTS}.csv')
        metrics_df.to_csv(metrics_file, index=False)

        # cc metrics
        metrics_dict = {
            'dataset': DATASET,
            'model': CHOSEN_MULTITASK_MODEL,
            'num_classes': NUM_CLASSES,
            'accuracy': cc_acc.item(),
            'precision_macro': cc_prec.item(),
            'recall_macro': cc_rec.item(),
            'f1_score_macro': cc_f1_score.item(),
            'total_samples': len(cc_true_labels)
        }

        per_class_precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average=None)
        per_class_recall = Recall(task="multiclass", num_classes=NUM_CLASSES, average=None)
        per_class_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average=None)

        prec_per_class = per_class_precision(cc_pred_labels, cc_true_labels)
        rec_per_class = per_class_recall(cc_pred_labels, cc_true_labels)
        f1_per_class = per_class_f1(cc_pred_labels, cc_true_labels)

        for i in range(NUM_CLASSES):
            metrics_dict[f'precision_class_{i}'] = prec_per_class[i].item()
            metrics_dict[f'recall_class_{i}'] = rec_per_class[i].item()
            metrics_dict[f'f1_score_class_{i}'] = f1_per_class[i].item()

        metrics_df = pd.DataFrame([metrics_dict])
        metrics_file = os.path.join(RESULTS_DIR, f'metrics({DATASET_NAME})(cc)_stage({STAGES_MAPPING[DATASET_STAGE]})_model({CHOSEN_MULTITASK_MODEL})_{TRAIN_WITH_WEIGHTS}_multiclass.csv')
        metrics_df.to_csv(metrics_file, index=False)

    else:
        raise Exception(f"Invalid option. Chose between 'subjectivity_classifier' and 'multitask_classifier'\nChosen task: {TASK}")


if __name__ == "__main__":
    main()