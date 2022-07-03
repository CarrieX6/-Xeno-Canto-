import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader,TensorDataset

class BIRDDATASET(object):

    def __init__(self, batch_size, use_gpu, num_workers):

        pin_memory = True if use_gpu else False
        data_features = np.load('data_CNN_feature_5s.npy')
        data_labels = np.load('data_CNN_label_5s.npy')

        data_labels = np.delete(data_labels, 0, axis=1)#[[1,0,0,0,0]]
        data_labels = np.argmax(data_labels, axis=1)

        X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, random_state=10)
        X_train = X_train.reshape(X_train.shape[0], 3, 40, 41)
        X_test = X_test.reshape(X_test.shape[0], 3, 40, 41)


        # trainloader =DataLoader(TensorDataset(torch.tensor(X_train).float(),torch.LongTensor(Y_train)), batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory)
        trainloader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.LongTensor(Y_train)),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        testloader =DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.LongTensor(Y_test)), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory)

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 5

__factory = {
    'birddataset': BIRDDATASET,
}

def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)
