import math
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader,TensorDataset
from cbam import CBAM

using_cbam = True


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.cbam1 = CBAM(out_planes, 2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        if using_cbam:
            out = self.cbam1(out)
        return out
class cnn(nn.Module):
    def __init__(self, nblocks=[6, 12, 24, 16], growth_rate=32, reduction=0.5, num_classes=5):
        super(cnn, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        num_planes += nblocks[0] * growth_rate
        self.linear1 = nn.Linear(num_planes, 2)
        self.prelu_fc1 = nn.PReLU()
        self.linear2 = nn.Linear(2, num_classes)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.prelu_fc1(self.linear1(x))
        y = self.linear2(x)
        return x,y


class DenseNet(nn.Module):
    def __init__(self, block=Bottleneck, nblocks=[6, 12, 24, 16], growth_rate=32, reduction=0.5, num_classes=5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        print(num_planes)
        self.linear1 = nn.Linear(num_planes, 2)
        self.prelu_fc1 = nn.PReLU()
        self.linear2 = nn.Linear(2, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = F.avg_pool2d(F.relu(self.bn(x)), 4)
        x = x .view(x .size(0), -1)

        x = self.prelu_fc1(self.linear1(x))
        y = self.linear2(x)
        return x,y

__factory = {
    'densenet121': DenseNet,
    'LSTM':cnn,

}
def create(name):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    # print(__factory[name])
    # exit()
    return __factory[name]


"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time


data_features = np.load('data_CNN_feature_5s.npy')
data_labels = np.load('data_CNN_label_5s.npy')

data_labels = np.delete(data_labels, 0, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, random_state=10)
X_train = X_train.reshape(X_train.shape[0], 3, 40, 41)
X_test = X_test.reshape(X_test.shape[0], 3, 40, 41)


trainloader =DataLoader(TensorDataset(torch.tensor(X_train).float(),torch.tensor(Y_train).float()),shuffle = True,batch_size = 256)
testloader =DataLoader(TensorDataset(torch.tensor(X_test).float(),torch.tensor(Y_test).float()),shuffle = False,batch_size =256)



 
#模型定义
model=DenseNet121()
# model=TeacherModel()
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()

# #################7.训练模型#####################
#开始训练
losses=[]
acces=[]
eval_losses=[]
eval_acces=[]
num_epochs=10
for epoch in range(num_epochs):
    train_loss=0
    train_acc=0
    model.train()

    for i,(img,label) in enumerate(trainloader):

        #前向传播
        out=model(img)

        loss=criterion(out,label)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        _, pred = out.max(1)

        label=np.argmax(label, axis=1)

        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc+=acc
    losses.append(train_loss/len(trainloader))
    acces.append(train_acc/len(trainloader))

    eval_loss=0
    eval_acc=0

    model.eval()
    for i,(img,label) in enumerate(testloader):

        out=model(img)
        loss=criterion(out,label)

        eval_loss+=loss.item()
        _,pred=out.max(1)
        label = np.argmax(label, axis=1)
        num_correct = (pred == label).sum().item()
        acc=num_correct/img.shape[0]
        print(acc)
        eval_acc+=acc

    eval_losses.append(eval_loss/len(testloader))
    eval_acces.append(eval_acc/len(testloader))
    print('epoch:{},Train Loss:{:.4f},Train Acc: {:.4f},Test Loss: {:.4f},Test Acc: {:.4f}'.
          format(epoch,train_loss/len(trainloader),train_acc/len(trainloader),
                 eval_loss/len(testloader),eval_acc/len(testloader)
                 ))
"""
