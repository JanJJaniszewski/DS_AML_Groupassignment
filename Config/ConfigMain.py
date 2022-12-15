from Config.Location import location

# Set model name and extract feature flag
# Available model names: [resnet18, resnet50, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet50"

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = True

# Set number of training epochs and batch size
num_epochs = 2
batch_size = 64

# Set mean and std values for normalization
means = [0.6178, 0.5074, 0.3895]
stds = [0.2437, 0.2478, 0.2533]

# Set learning rate and momentum
learning_rate = 0.001
momentum = 0.9

# Set location-specific configuration
if location == 'cinthy':
    pass
elif location == 'jesse':
    pass
elif location == 'sebas':
    pass
elif location == 'jan':
    pass
