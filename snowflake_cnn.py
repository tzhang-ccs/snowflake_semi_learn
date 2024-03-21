import torch
import torch.nn as nn
import sys
import os
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision
from loguru import logger
import argparse
import numpy as np

# # 1. CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fc1 = nn.Linear(32*56*56, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64,5)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0),-1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        
        return x

# # 2. data

class myDataset(Dataset):
    def __init__(self, data_path):
        n = 224
        transform = transforms.Compose([transforms.Resize((n,n)), transforms.ToTensor()])
        self.data = datasets.ImageFolder(f'{data_path}',transform)

    def __getitem__(self, index):
        x,y = self.data[index]
        y = torch.tensor(y)
        x = x.to(device)
        y = y.to(device)

        return x,y

    def __len__(self):
        self.num = len(self.data.targets)
        return self.num

def train():
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
    logger.add(sys.stdout, format=fmt)
    log_path = f'../saved_logs/cnn_log'
    if os.path.exists(log_path):
        os.remove(log_path)
    ii = logger.add(log_path)
    
    logger.debug(f'step 0: config')


    logger.debug(f'step 1: load data')
    #train_path = f'/pscratch/sd/z/zhangtao/storm/mpc/key_paper/training'
    train_path = f'/work/tzhang/storm/training'
    train_data = myDataset(train_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_len = train_data.num
    del train_data
    
    # # 3. train
    
    logger.debug(f'step 3: train')
    model = CNN()
    #model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002)
    
    for e in range(epoch):
        train_loss = 0.0
        for i,data in enumerate(train_loader):
            x,y = data
            optimizer.zero_grad()
    
            out = model(x)    
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
        
        logger.info(f'epoch={e}, loss = {train_loss/train_len:.5f}')
    
    torch.save(model, f'../saved_models/cnn')

def test():
    #test_path = f'/pscratch/sd/z/zhangtao/storm/mpc/key_paper/test/'
    test_path = f'/work/tzhang/storm/test/'
    test_data = myDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

    model = torch.load(f'../saved_models/cnn')

    total_correct = 0
    confusion_matrix = np.zeros([5,5],int)
    total_labels = np.zeros(5,int)

    with torch.no_grad():
        for data in test_loader:
            x,y = data
            out = model(x)
            _, out = torch.max(out.data,1)
            
            total_correct += (out == y).sum().item()
            for i,l in enumerate(y):
                confusion_matrix[l.item(),out[i].item()] += 1
    
            for ll in y:
                total_labels[ll] += 1


    accuracy = total_correct/test_data.num
    print(f'accuracy = {accuracy:.3f}')
    print(confusion_matrix/total_labels)

epoch = 100
batch_size = 128*4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--process", required=True)
parser.add_argument('-s', "--seed", required=True)

args = parser.parse_args()
process = args.process
seed = int(args.seed)

torch.manual_seed(seed)
np.random.seed(seed)


if process == 'train':
    train()
    test()

if process == 'test':
    test()


