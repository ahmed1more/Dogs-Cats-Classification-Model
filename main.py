import torch 
from torch import nn
import torch.optim as optim
from data_aug import train_loader, test_loader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from  arch import Cat_Dog_CNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from pathlib import Path


# Hyperparameters
learning_rate = 0.001
num_epochs = 20
# torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn = Cat_Dog_CNN().to(device)

##load a trianed
## p=Path('models/C&D_CNN_CLASSIFACTION_V9.pth')
## model_cnn.load_state_dict(torch.load(p))

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr= learning_rate)

# Initialize storage for metrics
train_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
test_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}



def training(model, loss_f, optim, epochs):
    best_model_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [] , []

        for images , labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_f(outputs,labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss+=loss.item()
            
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

        # Calculate metrics for the epoch
        acc_train = accuracy_score(all_labels, all_preds)
        precision_train = precision_score(all_labels, all_preds, average='weighted')
        recall_train = recall_score(all_labels, all_preds, average='weighted')
        f1_train = f1_score(all_labels, all_preds, average='weighted')

        train_metrics["loss"].append(train_loss / len(train_loader))
        train_metrics["accuracy"].append(acc_train)
        train_metrics["precision"].append(precision_train)
        train_metrics["recall"].append(recall_train)
        train_metrics["f1"].append(f1_train)

        # Save the best model
        if acc_train > best_model_acc:
            best_model_acc = acc_train
            Path("models").mkdir(parents=True, exist_ok=True)
            model_name = f'Best_Model.pth'
            model_path = Path(f"models/{model_name}")
            torch.save(obj=model.state_dict(), f=model_path)
            print(f"Saved best model to: {model_path}")
        
        testing(model,loss_f,train_loss,acc_train)


def testing(model, loss_f,train_loss,acc_train=None):
    model.eval()
    test_loss = 0
    all_preds, all_labels = [] , []

    with torch.inference_mode():
        for images , labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_f(outputs,labels)
            test_loss+=loss.item()
            
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    
    # Calculate metrics for testing
    acc_test = accuracy_score(all_labels, all_preds)
    precision_test = precision_score(all_labels, all_preds, average='weighted')
    recall_test = recall_score(all_labels, all_preds, average='weighted')
    f1_test = f1_score(all_labels, all_preds, average='weighted')

    test_metrics["loss"].append(test_loss / len(test_loader))
    test_metrics["accuracy"].append(acc_test)
    test_metrics["precision"].append(precision_test)
    test_metrics["recall"].append(recall_test)
    test_metrics["f1"].append(f1_test)

    print(f"Train Accuracy: {int(acc_train * 100)}% | Test Accuracy: {int(acc_test * 100)}%")
    print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")




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