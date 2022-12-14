import os

from Config.Location import location

if location == 'cinthy':
    training_data = './train_set'
    output_data = './Data/Output/Predictions_V{}.csv'
elif location == 'jesse':
    output_data = './Data/Output/Predictions_V{}.csv'
    input_path = './Data/Input/'
    input_train = input_path + 'train_set/train_set/train_set/'
    input_test = input_path + 'test_set/test_set/test_set/'
    input_labels_test = input_path + 'sample.csv'
    input_labels_train = input_path + 'train_labels.csv'
    output_data = './Data/Output/Predictions_V{}.csv'
    output_data_automatic = './Data/Output/Predictions_during_training_V{}.csv'
    throughput_path = './Data/Throughput/'
    A_trainset = throughput_path + 'A_folderize/train/'
    A_testset = throughput_path + 'A_folderize/test/'
    A_validationset = os.path.join(throughput_path, 'A_folderize/validation')

elif location == 'sebas':
    input_path = './Data/Input/'
    input_train = input_path + 'train_set/train_set/train_set/'
    input_test = input_path + 'test_set/test_set/test_set/'
    input_labels_test = input_path + 'sample.csv'
    input_labels_train = input_path + 'train_labels.csv'
    output_data = './Data/Output/Predictions_V{}.csv'
    output_data_automatic = './Data/Output/Predictions_during_training_V{}.csv'

    throughput_path = './Data/Throughput/'
    A_trainset = throughput_path + 'A_folderize/train/'
    A_testset = throughput_path + 'A_folderize/test/'
    A_validationset = os.path.join(throughput_path, 'A_folderize/validation')

elif location == 'jan':
    input_path = './Data/Input/'
    input_train = input_path + 'train_set/train_set/train_set/'
    input_test = input_path + 'test_set/test_set/test_set/'
    input_labels_test = input_path + 'sample.csv'
    input_labels_train = input_path + 'train_labels.csv'
    output_data = './Data/Output/Predictions_V{}.csv'
    output_data_automatic = './Data/Output/Predictions_during_training_V{}.csv'

    throughput_path = './Data/Throughput/'
    A_trainset = throughput_path + 'A_folderize/train'
    A_testset = throughput_path + 'A_folderize/test'
    A_validationset = os.path.join(throughput_path, 'A_folderize/validation')

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = A_trainset
