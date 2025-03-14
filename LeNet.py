import torch
import torch.nn as nn

# Credit to Haotong Liang (AuroraLHT)
class ModelOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits  # model results
        self.loss = loss  # loss function
        self.predictions = torch.flatten(torch.argmax(logits, dim=-1))

    def __str__(self):
        return str({'loss': self.loss, 'predictions': self.predictions, 'logits': self.logits})

    def accuracy(self, labels):
        assert labels.shape == self.predictions.shape, 'Pred and Labels don\'t have same shape!'
        accuracy = (torch.sum(self.predictions == labels) / len(self.predictions)).item()
        return round(accuracy, 4) * 100

    def top_k_preds(self, k):
        return torch.topk(self.logits, dim=-1, k=k).indices

    def top_k_acc(self, labels, k):
        labels = torch.unsqueeze(labels, dim=-1)
        labels = torch.cat([labels for _ in range(k)], dim=-1)
        labels = torch.unsqueeze(labels, dim=1)
        preds = self.top_k_preds(k)
        acc = (torch.sum(preds == labels) / len(labels)).item()
        return round(acc, 4) * 100
    
def get_flattened_size(model, input_size):
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, input_size)  # Batch size of 1, 1 channel, input size
        dummy_output = model.encoder(dummy_input)
        return dummy_output.view(-1).size(0)

import torch
class LeNet(nn.Module):
    def __init__(self, input_size,output_size):
        super(LeNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Determine flattened size
        self.flattened_size = self._get_flattened_size(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=output_size)
        )

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)  # Batch size of 1, 1 channel, input size
            dummy_output = self.encoder(dummy_input)
            return dummy_output.view(-1).size(0)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

