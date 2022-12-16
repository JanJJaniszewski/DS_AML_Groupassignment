from __future__ import print_function, division

import os
import re
import time

import pandas as pd
import torch
import torchvision
from torchvision import transforms

import Config.ConfigMain as conf
import Config.ConfigPaths as paths
import Functions.Utils as ut
from Config.Location import location


def A_Folderize(force=False):
    """Creates folders that are necessary in general and for training data.

    This function creates folders that are necessary in general and for training
    data, and then puts pictures in the newly created folders. The `force`
    parameter is optional and is set to `False` by default. If set to `True`,
    the function will re-create the folders even if they already exist.

    Args:
        force: bool, optional
            If set to `True`, the function will re-create the folders even if
            they already exist.

    Returns:
        None
    """
    print('START: A_folderize')
    print('Creating folders that are necessary in general')
    ut.make_folders()

    labels_test = pd.read_csv(paths.input_labels_test)
    labels_train = pd.read_csv(paths.input_labels_train)

    print('Creating folders that are necessary for training data')
    for label in labels_train['label'].unique():
        path = os.path.join(paths.A_trainset, str(label))
        os.mkdir(path)
    for label in labels_train['label'].unique():
        path = os.path.join(paths.A_validationset, str(label))
        os.mkdir(path)
    for label in labels_test['label'].unique():
        path = os.path.join(paths.A_testset, str(label))
        os.mkdir(path)

        print('Putting pictures in the newly created folders')
        for picpath in os.listdir(paths.input_train):  # picpath is the pathway to one image and the image its img_name,
            try: #look for the image
                labelfolder = labels_train.loc[labels_train['img_name'] == picpath]['label'].iloc[
                0]  # this returns a number indicating a foodclass
            except IndexError: #if image is not found, ignore since we deleted some filepaths    
                p = 6

            # Validation set: Everything that is dividable by 4, else training set -> (25% validation set)
            try:
                idx = int(re.findall(r'\d+', picpath)[0])
            except IndexError:
                idx = 4

            if (idx % 5) == 0:
                ut.copy_file(os.path.join(paths.input_train, picpath),
                             os.path.join(paths.A_validationset, str(labelfolder), picpath))
            else:
                ut.copy_file(os.path.join(paths.input_train, picpath),
                             os.path.join(paths.A_trainset, str(labelfolder), picpath))

        for picpath in os.listdir(paths.input_test):
            labelfolder = labels_test.loc[labels_test['img_name'] == picpath]['label'].iloc[0]
            ut.copy_file(os.path.join(paths.input_test, picpath),
                         os.path.join(paths.A_testset, str(labelfolder), picpath))

    print('DONE: A_folderize')


def B_InitModel():
    """Initializes a model.

    This function initializes a model and returns the initialized model and its
    input size.

    Returns:
        model: the initialized model
        input_size: the input size of the model
    """
    print('START: B_InitModel')
    classes = [name for name in os.listdir(paths.A_trainset) if os.path.isdir(os.path.join(paths.A_trainset, name))]
    num_classes = len(classes)
    print(f'Number of classes: {num_classes}')
    model = ut.initialize_model(conf.model_name, num_classes, conf.feature_extract)
    print('DONE: B_InitModel')
    return model


def C_PrepareData():
    """Prepares data for training, validation, and testing.

    This function prepares data for training, validation, and testing by
    applying transformations to the images in the datasets and loading the
    transformed images into data loaders.

    Args:
        input_size: int
            The input size of the model.

    Returns:
        train_loader: the data loader for the training dataset
        val_loader: the data loader for the validation dataset
        test_loader: the data loader for the testing dataset
    """

    print('START: C_PrepareData')
    input_size = 224

    # image data preparation
    # normalized data=> image=(image-mean)/std
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            # TODO: Check if better crop or transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),  # TODO: Check if resizing helps or not
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ]),
        # WARNING: KEEP SAME TO "val" transformer!!!!!!
        'test': transforms.Compose([
            transforms.Resize(input_size),  # TODO: Check if resizing helps or not
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ])
    }

    train_dataset = torchvision.datasets.ImageFolder(root=paths.A_trainset, transform=data_transforms['train'])
    val_dataset = torchvision.datasets.ImageFolder(root=paths.A_validationset, transform=data_transforms['val'])
    test_dataset = torchvision.datasets.ImageFolder(root=paths.A_testset, transform=data_transforms['test'])

    # Mini-Batch Gradient Descent, start with 32 and explore to increase performance
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)
    # print(ut.get_mean_and_std(train_loader))

    print('DONE: C_PrepareData')

    return train_loader, val_loader, test_loader


def D_TrainModel(model, train_loader, val_loader, test_loader):
    print('START: D_TrainModel')
    model, val_acc_history = ut.train_model(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_epochs=conf.num_epochs)
    print('DONE: D_TrainModel')
    return model, val_acc_history


def E_PredictModel(model, test_loader, verbose=0, version=int(time.time())):
    print("STARTED: Predictions")
    df= ut.predict_model(model, test_loader, verbose=verbose, version=version)
    print("DONE: Predictions")

    return df
