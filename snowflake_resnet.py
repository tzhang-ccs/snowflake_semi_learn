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
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler

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
    log_path = f'../saved_logs/resnet_log'
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
    resnet = models.resnet50(pretrained=True)
    num_features = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_features, 5)
    resnet = resnet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.002)
    
    resnet.train()

    for e in range(epoch):
        train_loss = 0.0
        for i,data in enumerate(train_loader):
            x,y = data
            optimizer.zero_grad()
    
            print("xx: ",x.shape,y.shape)

            out = resnet(x)    
            print("out: ", out.shape)
            loss = criterion(out,y)
            print("loss: ", loss)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
        
        logger.info(f'epoch={e}, loss = {train_loss/train_len:.5f}')
 
    os.system(f'rm -rf ../saved_models/resnet')
    torch.save(resnet, f'../saved_models/resnet')

def test():
    #test_path = f'/pscratch/sd/z/zhangtao/storm/mpc/key_paper/test/'
    test_path = f'/work/tzhang/storm/test/'
    test_data = myDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

    model = torch.load(f'../saved_models/resnet')
    model.eval()

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
batch_size = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--process", required=True)

args = parser.parse_args()
process = args.process

if process == 'train':
    train()

if process == 'test':
    test()


