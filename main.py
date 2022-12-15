from __future__ import print_function, division

import torch
import torchvision

import Functions.Pipeline as pipe

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def main(*pipeparts):
    if 'A' in pipeparts:
        pipe.A_Folderize(force=False)
    if 'B' in pipeparts:
        model = pipe.B_InitModel()
    if 'C' in pipeparts:
        train_loader, val_loader, test_loader = pipe.C_PrepareData()
    if 'D' in pipeparts:
        model, val_acc_history = pipe.D_TrainModel(model, train_loader, val_loader, test_loader)
    if 'E' in pipeparts:
        predictions = pipe.E_PredictModel(model, test_loader)
        return predictions


if __name__ == '__main__':
    # Define which parts of the pipeline to execute (include 'A' in first execution, then 'A' does not need to be run)
    pipeparts = ['B', 'C', 'D', 'E']
    main(*pipeparts)
