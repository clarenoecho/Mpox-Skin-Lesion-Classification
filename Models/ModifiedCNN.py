import torch
import torch.nn as nn


def _block(cin, cout, act):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        act(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        act(),
        nn.MaxPool2d(2)
    )


class ModifiedCNN(nn.Module):
    """Model 2: modifikasi Simple CNN."""
    input_size = 128

    def __init__(self, input_c=3, num_classes=6, dropout=0.5):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.1, inplace=True)
        self.features = nn.Sequential(
            _block(input_c, 32, act),
            _block(32, 64, act),
            _block(64, 128, act)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.fc(self.drop(torch.flatten(x, 1)))
