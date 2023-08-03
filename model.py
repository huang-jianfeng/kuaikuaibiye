import torch.nn as nn
import torch
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self) :
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 128, 3, 1, 1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, 2)) 
        self.dense = nn.Sequential(nn.Linear(14*14*128, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(p = 0.5),
                                         nn.Linear(1024, 10))
   
    def forward(self, x) :
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x

class MinistNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）

        
class Cifar10Net(torch.nn.Module):
        def __init__(self):
            super(Cifar10Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5) 
            self.conv2 = nn.Conv2d(6, 16, 5)  
            self.fc1   = nn.Linear(16*5*5, 120)  
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, 10)  # 最后是一个十分类，所以最后的一个全连接层的神经元个数为10

        def forward(self, x): 
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
            x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
            x = x.view(x.size()[0], -1)  # 展平  x.size()[0]是batch size
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)        
            return x
        

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def getVGG16():
    return VGG('VGG16')


class CNN(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_kernels, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=n_kernels, out_channels=2 * n_kernels, kernel_size=5)
        self.fc1 = nn.Linear(in_features=2 * n_kernels * 5 * 5, out_features=120)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(in_features=84, out_features=out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
