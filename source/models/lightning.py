import torch
import os
import json
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningDataModule, LightningModule
from torch import FloatTensor, UntypedStorage
from torch.nn import ModuleDict
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall, MetricCollection
from typing import Any, List, Tuple, TypeVar, Optional, Dict

def multitask_collate_fn(batch):
    """
    Collate function for multitask learning.
    Inputs: (embedding, labels_tuple)
    Returns: (embeddings_batch, (subj_labels_batch, cc_labels_batch))
    """
    embeddings = torch.stack([item[0] for item in batch])
    subj_labels = torch.stack([item[1] for item in batch])
    cc_labels = torch.stack([item[2] for item in batch])

    return embeddings, (subj_labels, cc_labels)

class DataModule(LightningDataModule):
    def __init__(self, *, datasets_root: str, dataset_id: int, batch_size: int = 1, num_workers: int = 6):
        super().__init__()
        self.datasets_root = datasets_root
        self.dataset_id = dataset_id
        self.labels_shape = dataset_id  # subj dataset (ID:1) has one label, cc dataset (ID:2) has two labels (multitask)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters() # saves hyperparameters in checkpoint file

    def _load_dataset(self, datasets_root, dataset_id, stage: str ):
        """
        Loads datasets accordingly with dataset_id:
        - ID 1: Single task (subjectivity)
        - ID 2: Multitask (subj + cc)
        """
        with open(os.path.join(datasets_root, 'metadata', f'metadata_{stage}_{dataset_id}.json'), "r") as file:
            metadata = json.load(file)
        total_sentences = metadata['total_sentences']
        embedding_size = metadata['embedding_size']
        embeddings = FloatTensor(torch.from_file(
                                                 os.path.join(datasets_root, f"embeddings_{stage}_{dataset_id}.bin"),
                                                              shared = False, size= (total_sentences * embedding_size))).view(total_sentences, embedding_size)
        # loads subj dataset
        if dataset_id == 1:
            labels = FloatTensor(
                torch.from_file(
                                os.path.join(datasets_root, f"labels_{stage}_{dataset_id}.bin"),
                                shared=False, size=total_sentences)).view(total_sentences, 1)
            return TensorDataset(embeddings, labels)
        # loads multitask dataset
        elif dataset_id == 2:
            subj_labels = FloatTensor(
                torch.from_file(
                                os.path.join(datasets_root, f"labels_subj_{stage}_{dataset_id}.bin"),
                                             shared=False,size=total_sentences ))

            cc_labels = FloatTensor(
                torch.from_file(
                                os.path.join(datasets_root, f"labels_cc_{stage}_{dataset_id}.bin"),
                                shared=False, size=total_sentences))

            return TensorDataset(embeddings, subj_labels, cc_labels)

        else:
            raise ValueError(f"Invalid dataset_id: {dataset_id}. Expected 1 (single task) or 2 (multitask)")


    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_subset = self._load_dataset(self.datasets_root, self.dataset_id, "train")
            self.val_subset = self._load_dataset(self.datasets_root, self.dataset_id, "val")
        elif stage == "validate":
            self.val_subset = self._load_dataset(self.datasets_root, self.dataset_id, "val")
        elif stage == "test":
            self.test_subset = self._load_dataset(self.datasets_root, self.dataset_id, "test")

    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True,
                          shuffle=True, drop_last=True, collate_fn=multitask_collate_fn if self.dataset_id == 2 else None)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True,
                          shuffle=False, drop_last=True, collate_fn=multitask_collate_fn if self.dataset_id == 2 else None)

    def test_dataloader(self):
        return DataLoader(self.test_subset,  batch_size=1, num_workers=self.num_workers, persistent_workers=True,
                          shuffle=False, drop_last=False, collate_fn=multitask_collate_fn if self.dataset_id == 2 else None)

class BaseModel(LightningModule):
    def __init__(self, *, model: torch.nn.Module, batch_size: int = 1, loss_function: torch.nn.Module, learning_rate: float= 0.01, learning_rate_patience: int = None, dataset_id: int = 1, **kwargs: Any):
        super().__init__(**kwargs)

        self.model = model()
        self.criterion = loss_function
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.dataset_id = dataset_id
        self.metrics = ModuleDict({mode_name: MetricCollection({  # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs
                                                            "acc": Accuracy(task="binary"),
                                                            "precision": Precision(task="binary"),
                                                            "recall": Recall(task="binary"),
                                                            "F1-score": F1Score(task="binary")
                                                            }) for mode_name in ["Train", "Val", "Test"]})

        assert self.dataset_id == 1, f"BaseModel is used for train single task models.\nSubjectivity Dataset (ID:1) is a single task model.\nClimate Change Dataset (ID:2) is a multitask model."

    def configure_optimizers(self):
        optmizer = SGD(self.parameters(), lr = self.learning_rate)
        scheduler = ReduceLROnPlateau(optmizer, mode='max', patience=self.learning_rate_patience)
        return {"optimizer": optmizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "recall/Val/Epoch"}}

    def forward(self, x: Any):
        return self.model(x)

    #defines basics operations for train, validadion and test
    def _any_step(self, batch: Tuple[torch.tensor, torch.tensor], stage: str):
        X, y = batch[0].squeeze(), batch[1].squeeze()
        predicted_value = self(X)
        predicted_value = predicted_value.squeeze()
        # Compute and log the loss value.
        loss = self.criterion(predicted_value, y)
        self.log(f"Loss/{stage}", loss, prog_bar=True)
        # Compute and log step metrics.
        y_int = y.int()
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Step': value for metric_name, value in metrics(predicted_value, y_int).items()})
        #print(predicted_value, y_int)
        return loss

    def training_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Train")

    def validation_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Val")

    def test_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Test")

    def _any_epoch_end(self, stage: str):
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Epoch': value for metric_name, value in metrics.compute().items()}, on_step=False, on_epoch=True) # logs metrics on epoch end
        metrics.reset()
        # Print loss at the end of each epoch
        #loss = self.trainer.callback_metrics[f"Loss/{stage}"]
        #print(f"Epoch {self.current_epoch} - Loss/{stage}: {loss.item()}")

    def on_train_epoch_end(self):
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self):
        self._any_epoch_end("Val")

    def on_test_epoch_end(self):
        self._any_epoch_end("Test")


    # def predict_step(self, batch, batch_idx):
    #     # change model to evaluation mode
    #     #model.eval()
    #     # variables
    #     partial_loss = []
    #     predicted_value = []
    #     expected_value = []
    #     #total_samples = len(eval_loader)
    #     # disable gradient calculation
    #     with torch.no_grad():
    #         for X_batch, y_batch in batch:

    #             y_pred = self.model(X_batch).squeeze(1)
    #             predicted_value.append(round(y_pred.item(), 2))

    #             expected_value.append(y_batch.item())

    #             loss = self.criterion(y_pred, y_batch)
    #             partial_loss.append(loss.item())

    #     partial_loss = np.array(partial_loss)
    #     predicted_value = np.array(predicted_value)
    #     expected_value = np.array(expected_value)

    #     return partial_loss, predicted_value, expected_value # ,accuracy

class MultitaskModel(LightningModule):
    def __init__(self, *, cc_model: torch.nn.Module, subj_trained_model: torch.nn.Module = None, subj_loss: torch.nn.Module, cc_loss: torch.nn.Module, batch_size: int = 1,learning_rate: float= 0.01, learning_rate_patience: int = None, dataset_id: int = 2, num_cc_classes: int = 18, **kwargs: Any):
        super().__init__(**kwargs)
        self.cc_model = cc_model()
        self.subj_trained_model = subj_trained_model
        self.criterion1 = subj_loss
        self.criterion2 = cc_loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.num_cc_classes = num_cc_classes

        self.metrics = {mode_name: {
                                    "subj": MetricCollection({
                                                            "acc": Accuracy(task="binary"),
                                                            "precision": Precision(task="binary"),
                                                            "recall": Recall(task="binary"),
                                                            "F1-score": F1Score(task="binary")
                                    }),
                                    "cc": MetricCollection({
                                                            "acc": Accuracy(task="multiclass", num_classes=self.num_cc_classes),
                                                            "precision": Precision(task="multiclass", num_classes=self.num_cc_classes),
                                                            "recall": Recall(task="multiclass", num_classes=self.num_cc_classes),
                                                            "F1-score": F1Score(task="multiclass", num_classes=self.num_cc_classes)
                                    }),
                                    } for mode_name in ["Train", "Val", "Test"]}

        assert dataset_id == 2, f"MultitaskModel is used for train multitask models.\nSubjectivity Dataset (ID:1) is a single task model.\nClimate Change Dataset (ID:2) is a multitask model."

    def configure_optimizers(self):
        parameters = list(self.subj_trained_model.parameters()) + list(self.cc_model.parameters()) #Autograd take cares to calculate the gradient for each paramater. Therefore, optmizing each model. The optimizer doesn't knows (and doesnt need to) from which models came the parameters
        optimizer = SGD(parameters, lr = self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=self.learning_rate_patience)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "Total_Loss/Val", "interval": "epoch", "frequency": 1,}}

    def forward(self, x: Any):
        subj_prediction = self.subj_trained_model(x)
        cc_prediction = self.cc_model(x)
        return subj_prediction, cc_prediction

    def L2_reg(self, model1_params: torch.nn, model2_params: torch.nn, weight_decay:float):
        # 1. Crie um dicionário (nome -> parâmetro) para o modelo 2
        #    para busca rápida.
        params2 = dict(model2_params)

        l2_reg = 0.0

        # 2. Itere sobre o modelo 1 (nome, parâmetro)
        for name1, param1 in model1_params:

            # 3. Verifique se o modelo 2 TAMBÉM TEM um parâmetro
            #    com o MESMO NOME
            if name1 in params2:
                param2 = params2[name1]

                # 4. VERIFIQUE SE AS FORMAS (SHAPES) SÃO IGUAIS!
                #    Esta é a verificação que evita o seu erro.
                if param1.shape == param2.shape:

                    # 5. Só se tudo bater, aplique a penalidade
                    l2_reg += torch.sum((param1 - param2) ** 2)

        return weight_decay * l2_reg

    #defines basics operations for train, validadion and test
    def _any_step(self, batch: Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]], stage: str):
        X, (y1, y2) = batch
        subj_prediction, cc_prediction = self.subj_trained_model(X), self.cc_model(X)
        # computes L2 regularization for parameters of subj_model e cc_model
        L2_regularization = self.L2_reg(self.subj_trained_model.named_parameters(), self.cc_model.named_parameters(), weight_decay=0.001)
        # Compute and log the loss value.
        loss1 = self.criterion1(subj_prediction.squeeze(), y1)
        loss2 = self.criterion2(cc_prediction, y2.long())
        loss = loss1 + loss2 + L2_regularization
        self.log(f"Loss_subj_model/{stage}", loss1, prog_bar=False)
        self.log(f"Loss_cc_model/{stage}", loss2, prog_bar=False)
        self.log(f"Weight_decay/{stage}", L2_regularization, prog_bar=False)
        self.log(f"Total_Loss/{stage}", loss, prog_bar=True)
        # Predictions
        subj_pred_labels = (torch.sigmoid(subj_prediction) > 0.5).int()
        cc_pred_labels = cc_prediction.argmax(dim=1)
        # Compute and log step metrics.
        subj_metrics = self.metrics[stage]["subj"](subj_pred_labels.squeeze(), y1.int())
        cc_metrics = self.metrics[stage]["cc"](cc_pred_labels, y2.int())
        subj_log = {f"{metric}/subj/{stage}/Step": value for metric, value in subj_metrics.items()}
        cc_log   = {f"{metric}/cc/{stage}/Step": value for metric, value in cc_metrics.items()}
        self.log_dict(subj_log)
        self.log_dict(cc_log)
        print(loss1, loss2, L2_regularization)
        return loss

    def training_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Train")

    def validation_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Val")

    def test_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Test")

    def _any_epoch_end(self, stage: str):
        subj_metrics = self.metrics[stage]["subj"]
        cc_metrics = self.metrics[stage]["cc"]
        # Compute & log subjectivity metrics
        subj_computed = subj_metrics.compute()
        subj_log = {f'{metric_name}/subj/{stage}/Epoch': value for metric_name, value in subj_computed.items()}
        self.log_dict(subj_log, on_step=False, on_epoch=True)
        # Compute & log climate change metrics
        cc_computed = cc_metrics.compute()
        cc_log = {f'{metric_name}/cc/{stage}/Epoch': value for metric_name, value in cc_computed.items()}
        self.log_dict(cc_log, on_step=False, on_epoch=True)
        # Reset metrics
        subj_metrics.reset()
        cc_metrics.reset()

    def on_train_epoch_end(self):
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self):
        self._any_epoch_end("Val")

    def on_test_epoch_end(self):
        self._any_epoch_end("Test")
