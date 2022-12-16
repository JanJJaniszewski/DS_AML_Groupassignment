from __future__ import division
from __future__ import print_function
from __future__ import print_function, division

import copy
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

import Config.ConfigMain as conf
import Config.ConfigPaths as paths
from Config.Location import location


def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    mean /= total_images_count
    std /= total_images_count
    return mean, std

def define_model(model):
    """Modify the fully-connected layer of a PyTorch model to match the number of features and classes in the input data.

    Args:
        model (torch.nn.Module): The PyTorch model to modify.

    Returns:
        torch.nn.Module: The modified PyTorch model.
    """
    num_fts = model.fc.in_features
    number_of_classes = len(pd.read_csv(paths.input_labels_train)['label'].unique())

    model.fc = nn.Linear(num_fts, number_of_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(torch.cuda.is_available())
    return model


def init_optimizer(model_ft, device='cpu'):
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if name in ['layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'fc.weight', 'fc.bias']:
            params_to_update.append(param)
            param.requires_grad = True
            print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=conf.learning_rate, momentum=conf.momentum)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2")

    return optimizer, scheduler


def predict(model, test_loader):
    pass


def evaluate_model_on_test_set(model, test_loader):
    pass


def show_transf_images(data):
    loader = torch.utils.data.DataLoader(data, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels:', labels)


def make_folders():
    for path in [paths.A_trainset, paths.A_testset, paths.A_validationset]:
        if not os.path.exists(path):
            os.mkdir(path)


def copy_file(source, destination):
    shutil.copyfile(source, destination)


def train_model(model, optimizer, scheduler, dataloaders, num_epochs=conf.num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    version = int(time.time())
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Print the model we just instantiated
    print(f"Model: {model}")
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    assert len(dataloaders['val'].dataset) < len(dataloaders['train1'].dataset), 'ERROR: Validation datasets >= Training dataset'

    dataloaders['val2'] = dataloaders['val']
    dataloaders['val3'] = dataloaders['val']

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")


        # Each epoch has a training and validation phase
        for phase in ['train3', 'val3', 'train2', 'val2', 'train1', 'val']:
            is_train = 'train' in phase
            if is_train:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(is_train):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if is_train:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts  = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        if epoch_acc >= best_acc:
            with torch.set_grad_enabled(False):
                predict_model(model, dataloaders['test'], version=version, filepath = paths.output_data_automatic)

        scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def find_class(idxz):
    classes = [name for name in os.listdir(paths.A_trainset) if os.path.isdir(os.path.join(paths.A_trainset, name))]
    classes.sort()
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    retclass = idx_to_class[idxz]
    return retclass



def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 224

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        Exception(
            "Invalid model name, exiting. To change model name, change the model_name variable in Config.ConfigMain")
        exit()

    return model_ft


def plot_model():
    # TODO: Not finished yet
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    hist = []
    ohist = [h.cpu().numpy() for h in hist]
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, conf.num_epochs + 1), ohist, label="Pretrained")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, conf.num_epochs + 1, 1.0))
    plt.legend()
    plt.show()


def predict_model(model, test_loader, verbose=0, version=int(time.time()), filepath = paths.output_data):
    model.eval()

    # Image names
    imagenames = []
    for data in test_loader.dataset.imgs:
        imgname = data[0].split("/")[-1]
        if location in ['sebas', 'cynthia', 'jesse']:
            imgname = imgname[2:]
        imagenames.append(imgname)

    predictions = []
    for inputs, _ in test_loader:
        inputs = inputs.to('cpu')
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = [find_class(int(idx)) for idx in preds]
        predictions += preds

    df = pd.DataFrame({
        "img_name": imagenames,
        "label": predictions
    })
    if verbose > 1:
        print(df)
    df.to_csv(filepath.format(version), index=False)

    return df
