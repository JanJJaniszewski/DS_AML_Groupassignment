from __future__ import print_function, division

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

import Config.ConfigMain as conf
import Config.ConfigPaths as paths


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


def set_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'

    return dev


def define_model(model=conf.model_main):
    num_fts = model.fc.in_features
    number_of_classes = len(pd.read_csv(paths.input_labels_train)['label'].unique())

    model.fc = nn.Linear(num_fts, number_of_classes)
    device = set_device()
    model = model.to(device)

    return model


def train_nn(model, train_loader, test_loader, criterion=conf.loss_fn, optimizer=conf.optimizer,
             n_epochs=conf.n_epochs):
    device = set_device()

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_correct / total

        print(f"Epoch {epoch + 1}: Correct: {running_correct} ({epoch_acc}%); Loss: {epoch_loss}")


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
    for path in [paths.A_trainset, paths.A_testset]:
        if not os.path.exists(path):
            os.mkdir(path)


copy_file = lambda src_file, dest: os.system(f"cp {src_file} {dest}")
