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
            try:
                labelfolder = labels_train.loc[labels_train['img_name'] == picpath]['label'].iloc[0]  # this returns a number indicating a foodclass
            except IndexError:
                pass

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = [name for name in os.listdir(paths.A_trainset) if os.path.isdir(os.path.join(paths.A_trainset, name))]
    num_classes = len(classes)
    print(f'Number of classes: {num_classes}')
    model = ut.initialize_model(conf.model_name, num_classes, conf.feature_extract)
    # Observe that all parameters are being optimized
    optimizer, scheduler = ut.init_optimizer(model, device)


    print('DONE: B_InitModel')
    return model, optimizer, scheduler


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
        'train1': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ]),
        'train2': transforms.Compose([
            transforms.AutoAugment(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ]),
        'train3': transforms.Compose([
            transforms.RandAugment(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ]),
        # WARNING: KEEP SAME TO "val" transformer!!!!!!
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(conf.means, conf.stds)
        ])
    }

    train_dataset1 = torchvision.datasets.ImageFolder(root=paths.A_trainset, transform=data_transforms['train1'])
    train_dataset2 = torchvision.datasets.ImageFolder(root=paths.A_trainset, transform=data_transforms['train2'])
    train_dataset3 = torchvision.datasets.ImageFolder(root=paths.A_trainset, transform=data_transforms['train3'])
    val_dataset = torchvision.datasets.ImageFolder(root=paths.A_validationset, transform=data_transforms['val'])
    test_dataset = torchvision.datasets.ImageFolder(root=paths.A_testset, transform=data_transforms['test'])

    # Mini-Batch Gradient Descent, start with 32 and explore to increase performance
    train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=conf.batch_size, shuffle=True)
    train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=conf.batch_size, shuffle=True)
    train_loader3 = torch.utils.data.DataLoader(train_dataset3, batch_size=conf.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

    # print(ut.get_mean_and_std(train_loader1))

    loaders = {
        'train1': train_loader1,
        'train2': train_loader2,
        'train3': train_loader3,
        'val': val_loader,
        'test': test_loader
    }

    # for phase, loader in loaders.items():
    #     assert all([sum(d[0].shape) == (224+224+3) for d in loader.dataset]), f'Not all pictures have the same size for {phase} loader'

    print('DONE: C_PrepareData')
    return loaders


def D_TrainModel(model, optimizer, scheduler, loaders):
    print('START: D_TrainModel')
    model, val_acc_history = ut.train_model(model, optimizer, scheduler, dataloaders=loaders, num_epochs=conf.num_epochs)
    print('DONE: D_TrainModel')
    return model, val_acc_history


def E_PredictModel(model, test_loader, verbose=0, version=int(time.time())):
    print("STARTED: Predictions")
    df = ut.predict_model(model, test_loader, verbose=verbose, version=version)
    print("DONE: Predictions")

    return df