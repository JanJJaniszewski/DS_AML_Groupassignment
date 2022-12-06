from Config.Location import location

if location == 'cinthy':
    training_data = './train_set'
elif location == 'jesse':
    pass
elif location == 'sebas':
    input_path = './Data/Input/'
    input_train = input_path + 'train_set/train_set/train_set/'
    input_test = input_path + 'test_set/test_set/test_set/'
    input_labels_test = input_path + 'sample.csv'
    input_labels_train = input_path + 'train_labels.csv'

    throughput_path = './Data/Throughput/'
    A_trainset = throughput_path + 'A_folderize/train/'
    A_testset = throughput_path + 'A_folderize/test/'
elif location == 'jan':
    input_path = './Data/Input/'
    input_train = input_path + 'train_set/train_set/train_set/'
    input_test = input_path + 'test_set/test_set/test_set/'
    input_labels_test = input_path + 'sample.csv'
    input_labels_train = input_path + 'train_labels.csv'

    throughput_path = './Data/Throughput/'
    A_trainset = throughput_path + 'A_folderize/train/'
    A_testset = throughput_path + 'A_folderize/test/'
