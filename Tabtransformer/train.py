import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from tabtransformer import *

import time
start_time = time.time()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
path_train = "../3D_face/BU_3DFE/features_3d_meshes/G0_train.csv"
path_test  = "../3D_face/BU_3DFE/features_3d_meshes/G0_test.csv"

path_other_test = "/3D_face/FaceScape/features_3d_meshes/G0_test.csv"

# path_save_model = "..../Model_bu_G0.pth" 

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Define custom dataset class
class LoadDataset(Dataset):
    def __init__(self, filepath, scaler=None):
        self.data = pd.read_csv(filepath)
        self.data = self.data.drop(["id", "name", "source"], axis=1)
        self.X    = self.data.drop("label", axis=1).values
        self.y    = self.data["label"].values
        self.scaler = scaler
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.X      = self.scaler.fit_transform(self.X)
        else:
            self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    
    def get_scaler(self):
        return self.scaler
           
#-------------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # Change to float for BCELoss

            optimizer.zero_grad()
            outputs = model(batch_x)

            # Ensure outputs are float and squeeze to match batch_y dimensions
            outputs = outputs.float().squeeze()

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} : Train Loss : {train_loss:.4f}")
        print(f"Val Loss : {val_loss:.4f}, Val Accuracy : {val_accuracy:.4f}")
        print("-" * 50)

#-------------------------------------------------------------------------------
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # Change to float for BCELoss

            outputs = model(batch_x)
            outputs = outputs.float().squeeze()

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()  # Use 0.5 as threshold for binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

#-------------------------------------------------------------------------------
def get_metrics(model, test_loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)

            outputs = model(batch_x)
            preds   = (outputs.squeeze() > 0.5).float()  
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall    = recall_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1

#-------------------------------------------------------------------------------   
#-------------------------------------------------------------------------------
# Load dataset
print("Load dataset ...")
train_dataset = LoadDataset(filepath=path_train, scaler=None)
my_scaler     = train_dataset.get_scaler()
test_dataset  = LoadDataset(filepath=path_test, scaler=my_scaler)

other_dataset = LoadDataset(filepath=path_other_test, scaler=my_scaler)

# Data loaders
batch_size   = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

other_test_loader  = DataLoader(other_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, loss function, and optimizer
num_epochs   = 300

num_features = train_dataset.X.shape[1]
d_model      = 128
num_heads    = 4
num_layers   = 3
d_ff         = 256
num_classes  = 1  # binary classification
dropout_rate = 0.1

model        = TabTransformer(num_features, d_model, num_heads, num_layers, d_ff, num_classes, dropout_rate)

lr           = 3e-4
weight_decay = 0.3
amsgrad      = "store_true"
optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
criterion    = nn.BCELoss() 

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------
# Train the model
print("Start training ...")
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

# Evaluate the model on the testset
print("-" * 50)
print("Evaluate the model : ")
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Get additional metrics (1)
accuracy, precision, recall, f1 = get_metrics(model, test_loader, device)
print(f"Precision: {precision:.4f} : Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

#-------------------------------------------------------------------------------
# Evaluate the model on another testset (other_test_loader)
print("-" * 50)
print("other_test_loader : ")
test_loss, test_accuracy = evaluate_model(model, other_test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Get additional metrics (2)
accuracy, precision, recall, f1 = get_metrics(model, other_test_loader, device)
print(f"Precision: {precision:.4f} : Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("-" * 50)

#-------------------------------------------------------------------------------
# Save the model state dictionary
#torch.save(model.state_dict(), path_save_model)

print("-" * 50)
end_time = time.time()
print("The execution time (in seconds) is : ",end_time-start_time)
print("Done ...")

