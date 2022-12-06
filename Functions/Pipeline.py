from __future__ import print_function, division

import os

import pandas as pd
import torch
import torchvision
from torchvision import transforms

import Config.ConfigMain as conf
import Config.ConfigPaths as paths
import Functions.Utils as ut


def A_Folderize():
    """
    Moving all pictures into separate folders
    :return:
    """
    print('START: A_folderize')
    print('Creating folders that are necessary in general')
    ut.make_folders()

    labels_test = pd.read_csv(paths.input_labels_test)
    labels_train = pd.read_csv(paths.input_labels_train)

    print('Creating folders that are necessary for training data')
    for label in labels_train['label'].unique():
        path = paths.A_trainset + str(label)
        if not os.path.exists(path):
            os.mkdir(path)
    for label in labels_test['label'].unique():
        path = paths.A_testset + str(label)
        if not os.path.exists(path):
            os.mkdir(path)

    print('Putting pictures in the newly created folders')
    for picpath in os.listdir(paths.input_train):
        labelfolder = labels_train.loc[labels_train['img_name'] == picpath]['label'].iloc[0]
        ut.copy_file(paths.input_train + picpath, paths.A_trainset + str(labelfolder))
    for picpath in os.listdir(paths.input_test):
        labelfolder = labels_test.loc[labels_test['img_name'] == picpath]['label'].iloc[0]
        ut.copy_file(paths.input_test + picpath, paths.A_testset + str(labelfolder))

    print('DONE: A_folderize')


def B_PrepareData(resize=conf.resize, means=conf.means, stds=conf.stds):
    """

    :param resize: Size of resized images
    :param means: means found for normalization
    :param stds: stds found for normalization
    :return: Train dataset, Test dataset
    """
    print('START: Prepare data for model')
    # resize, pixels. resize depending on data, we have to explore other sizes to check for performance
    training_transforms = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])
    print(training_transforms)

    train_dataset = torchvision.datasets.ImageFolder(root=paths.A_trainset, transform=training_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    # image data preparation
    # normalized data=> image=(image-mean)/std

    train_transforms = transforms.Compose([

        transforms.Resize((resize, resize)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(torch.Tensor(means), torch.Tensor(stds))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((resize, resize)),

        transforms.ToTensor(),

        transforms.Normalize(torch.Tensor(means), torch.Tensor(stds))
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=paths.A_trainset, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=paths.A_testset, transform=train_transforms)

    # Mini-Batch Gradient Descent, start with 32 and explore to increase performance
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    print('DONE: Prepare data for model')

    return train_loader, test_loader


def C_TrainModel(train_loader, test_loader):
    model = ut.define_model(model=conf.model_main)
    model = ut.train_nn(model, train_loader, test_loader, criterion=conf.loss_fn, optimizer=conf.optimizer,
                        n_epochs=conf.n_epochs)
    return model


def D_EvaluateModel():
    pass


def E_PredictModel():
    pass
