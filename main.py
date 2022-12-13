from __future__ import print_function, division

import Functions.Pipeline as pipe
import torch
import torchvision

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def main(*pipeparts):
    if 'A' in pipeparts:
        pipe.A_Folderize(force=False)
    if 'B' in pipeparts:
        model, input_size = pipe.B_InitModel()
    if 'C' in pipeparts:
        train_loader, val_loader, test_loader = pipe.C_PrepareData(input_size)
    if 'D' in pipeparts:
        model = pipe.D_TrainModel(model, train_loader, val_loader)
    if 'E' in pipeparts:
        print(pipe.E_PredictModel(model, test_loader))


if __name__ == '__main__':
    # Define which parts of the pipeline to execute (include 'A' in first execution, then 'A' does not need to be run)
    pipeparts = ['B', 'C', 'E']
    main(*pipeparts)
