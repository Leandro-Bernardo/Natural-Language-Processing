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



class DataModule(LightningDataModule):
    def __init__(self, *, datasets_root: str, batch_size: int , num_workers: int, dataset_id: int):
        super().__init__()
        self.datasets_root = datasets_root
        self.dataset_id = dataset_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters() # saves hyperparameters in checkpoint file

    def _load_dataset(self, datasets_root, dataset_id, stage: str ):
        with open(os.path.join(datasets_root, 'metadata', f'metadata_{stage}_{dataset_id}.json'), "r") as file:
            metadata = json.load(file)
        total_sentences = metadata['total_sentences']
        embedding_size = metadata['embedding_size']
        embeddings = FloatTensor(torch.from_file(os.path.join(datasets_root, f"embeddings_{stage}_{dataset_id}.bin"), shared = False, size= (total_sentences * embedding_size))).view(total_sentences, embedding_size)
        labels = FloatTensor(torch.from_file(os.path.join(datasets_root, f"labels_{stage}_{dataset_id}.bin"), shared = False, size= (total_sentences))).view(total_sentences)

        return TensorDataset(embeddings, labels)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_subset = self._load_dataset(self.datasets_root, self.dataset_id, "train")
            self.val_subset = self._load_dataset(self.datasets_root, self.dataset_id, "val")
        elif stage == "validate":
            self.val_subset = self._load_dataset(self.datasets_root, self.dataset_id, "val")
        elif stage == "test":
            self.test_subset = self._load_dataset(self.datasets_root, self.dataset_id, "test")

    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_subset,  batch_size=1, num_workers=self.num_workers, persistent_workers=True, shuffle=False, drop_last=True)

class BaseModel(LightningModule):
    def __init__(self, *, model: torch.nn.Module, batch_size: int, loss_function: torch.nn.Module, learning_rate: float, learning_rate_patience: int = None, dataset_id: int = None, **kwargs: Any):
        super().__init__(**kwargs)

        self.model = model()
        self.criterion = loss_function
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        if dataset_id == 1:
            self.metrics = ModuleDict({mode_name: MetricCollection({  # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs
                                                                "acc": Accuracy(task="binary"),
                                                                "precision": Precision(task="binary"),
                                                                "recall": Recall(task="binary"),
                                                                "F1-score": F1Score(task="binary")
                                                               }) for mode_name in ["Train", "Val", "Test"]})
        elif dataset_id == 2:
            self.metrics = ModuleDict({mode_name: MetricCollection({  # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs
                                                                "acc": Accuracy(task="multiclass", num_classes= 5),
                                                                "precision": Precision(task="multiclass", num_classes= 5),
                                                                "recall": Recall(task="multiclass", num_classes= 5),
                                                                "F1-score": F1Score(task="multiclass", num_classes= 5)
                                                               }) for mode_name in ["Train", "Val", "Test"]})
        #self.early_stopping_patience = early_stopping_patience

    def configure_optimizers(self):
        self.optimizer = SGD(self.parameters(), lr = self.learning_rate)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.learning_rate_patience)

        #return optimizer
        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.reduce_lr_on_plateau, "monitor": "Loss/Val"}}
        #return [self.optmizer], [self.reduce_lr_on_plateau]

    def forward(self, x: Any):
        return self.model(x)

    #defines basics operations for train, validadion and test
    def _any_step(self, batch: Tuple[torch.tensor, torch.tensor], stage: str):
        X, y = batch[0].squeeze(), batch[1].squeeze()
        predicted_value = self(X)    # o proprio objeto de BaseModel é o modelo (https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
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
