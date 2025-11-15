import torch.nn as nn
import torch

class multitask_model_v1(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.input_layer = nn.Sequential(
                                nn.Linear(in_features=384, out_features=128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.layer_1 = nn.Sequential(
                                nn.Linear(in_features=128, out_features=64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.classifier_layer = nn.Sequential(
                                    nn.Linear(in_features=64, out_features=self.num_classes),
                                    #nn.Softmax()  # already on loss torch.nn.CrossEntropyLoss()
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.classifier_layer(x)
        return x

class multitask_model_v2(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.input_layer = nn.Sequential(
                                nn.Linear(in_features=384, out_features=256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.layer_1 = nn.Sequential(
                                nn.Linear(in_features=256, out_features=128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.layer_2 = nn.Sequential(
                                nn.Linear(in_features=128, out_features=64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.layer_3 = nn.Sequential(
                                nn.Linear(in_features=64, out_features=32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.classifier_layer = nn.Sequential(
                                    nn.Linear(in_features=32, out_features=self.num_classes),
                                    #nn.Softmax()  # already on loss torch.nn.CrossEntropyLoss()
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.classifier_layer(x)
        return x

class multitask_model_v3(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.input_layer = nn.Sequential(
                                nn.Linear(in_features=384, out_features=128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.layer_1 = nn.Sequential(
                                nn.Linear(in_features=128, out_features=32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.classifier_layer = nn.Sequential(
                                    nn.Linear(in_features=32, out_features=self.num_classes),
                                    #nn.Sigmoid()  # already on loss torch.nn.BCEWithLogitsLoss()
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.classifier_layer(x)
        return x