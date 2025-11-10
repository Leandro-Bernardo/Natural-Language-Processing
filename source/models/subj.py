import torch.nn as nn
import torch

class subj_classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.layer_2 = nn.Sequential(
                                nn.Linear(in_features=64, out_features=16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        self.classifier_layer = nn.Sequential(
                                    nn.Linear(in_features=16, out_features=1),
                                    #nn.Sigmoid()  # already on loss torch.nn.BCEWithLogitsLoss()
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.classifier_layer(x)
        return x