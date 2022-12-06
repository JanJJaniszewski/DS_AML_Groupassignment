import torch.nn as nn
from torch import optim
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torch
from Config.Location import location

# Generali Config
# Model
model_main = models.resnet18(weights=ResNet18_Weights.DEFAULT)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 100
optimizer = optim.SGD(model_main.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)
resize = 224
means = [0.6366, 0.5437, 0.4454]
stds = [0.2235, 0.2422, 0.2654]

# Local config
if location == 'cinthy':
    pass
elif location == 'jesse':
    pass
elif location == 'sebas':
    pass
elif location == 'jan':
    pass
