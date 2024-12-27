import torch 
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms ,datasets
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 20
# torch.manual_seed(42)

transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize]) 

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])

train_dataset = datasets.ImageFolder('training_set',train_transforms)
test_dataset = datasets.ImageFolder('test_set',test_transforms)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)



# Model
class Cat_Dog_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5)
        self.fc1 = nn.Linear(128*10*10, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 128*10*10)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


################
##load a trianed
## p=Path('models/C&D_CNN_CLASSIFACTION_V9.pth')
## model_cnn.load_state_dict(torch.load(p))
##             
###############
model_cnn = Cat_Dog_CNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr= learning_rate)



def training(model, loss_f, optim, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [] , []

        for images , labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            
            outputs = model(images)
            loss = loss_f(outputs,labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss+=loss.item()
            
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            acc_train = accuracy_score(all_labels,all_preds)

        Path("models").mkdir(parents=True, exist_ok=True)
        model_name =f'C&D_CNN_CLASSIFACTION_V{epoch+1}.pth'
        model_path = Path(f"models/{model_name}")

        print(f"Saving the model to : {model_path}")
        torch.save(obj=model_cnn.state_dict(), f=model_path)
        
        testing(model,loss_f,train_loss,acc_train)


def testing(model, loss_f,train_loss,acc_train=None):
    model.eval()
    test_loss = 0
    all_preds, all_labels = [] , []

    with torch.inference_mode():
        for images , labels in test_loader:
            
            outputs = model(images)
            loss = loss_f(outputs,labels)
            test_loss+=loss.item()
            
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            acc_test = accuracy_score(all_labels,all_preds)

    print(f" accurcy_train: {int(acc_train*100)}% | accurcy_test: {int(acc_test*100)}% | training loss: {train_loss} | testing loss: | {test_loss}")




if __name__ == '__main__':
    ############
    ##Training##
    ############
    training(model_cnn,loss_function,optimizer,num_epochs)

    # Saving the model
    Path("models").mkdir(parents=True, exist_ok=True)
    model_name = 'C&D_CNN_CLASSIFACTION_PRIME_V1.pth'
    model_path = Path(f"models/{model_name}")

    print(f"Saving the model to : {model_path}")
    torch.save(obj=model_cnn.state_dict(), f=model_path)