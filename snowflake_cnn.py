import torch
import torch.nn as nn
import sys
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision

# # 1. CNN model
epoch = 500
batch_size = 128*4

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
        self.data = datasets.ImageFolder(f'{train_path}',transform)

    def __getitem__(self, index):
        x,y = self.data[index]
        y = torch.tensor(y)
        x = x.to(device)
        y = y.to(device)

        return x,y

    def __len__(self):
        self.len = len(self.data.targets)
        return self.len



print(f'step 1: load data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = f'/work/tzhang/storm/training'
train_data = myDataset(train_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# # 3. train

# In[99]:

print(f'step 3: train')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# In[ ]:


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
    
    print(f'epoch={e}, loss = {train_loss/train_data.len:.5f}')




