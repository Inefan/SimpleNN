import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import os
import matplotlib.pyplot as plt

import numpy as np

import json
from tqdm import tqdm

from PIL import Image
from torchvision import transforms

plt.style.use("dark_background")


device = "cuda" if torch.cuda.is_available() else "cpu"




class MNISTDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data_list = []
        
        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)  
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                continue
            cls = os.path.basename(path_dir)  
            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = Image.open(file_path).convert("L")
        if self.transform:
            sample = self.transform(sample)
        return sample, target

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


train_data = MNISTDataset("./mnist/training", transform=transform)
test_data = MNISTDataset("./mnist/testing", transform=transform)
train_data, val_data = random_split(train_data, [int(0.7 * len(train_data)), int(0.3 * len(train_data))])




train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)






class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, output_dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        return self.layer_2(x)




model = MyModel(784, 10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)


EPOCHS = 30
train_loss, train_acc, val_loss, val_acc, lr_list = [], [], [], [], []
best_loss = float("inf")




for epoch in range(EPOCHS):
    model.train()
    running_train_loss, correct_train = [], 0
    for x, targets in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, targets = x.to(device), targets.to(device)
        x = x.view(x.size(0), -1)
        preds = model(x)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item())
        correct_train += (preds.argmax(dim=1) == targets).sum().item()
    
    train_loss.append(np.mean(running_train_loss))
    train_acc.append(correct_train / len(train_data))
    
    model.eval()
    with torch.no_grad():
        running_val_loss, correct_val = [], 0
        for x, targets in val_loader:
            x, targets = x.to(device), targets.to(device)
            x = x.view(x.size(0), -1)
            preds = model(x)
            loss = loss_fn(preds, targets)
            running_val_loss.append(loss.item())
            correct_val += (preds.argmax(dim=1) == targets).sum().item()
        
        val_loss_epoch = np.mean(running_val_loss)
        val_acc.append(correct_val / len(val_data))
        val_loss.append(val_loss_epoch)
        lr_scheduler.step(val_loss_epoch)
        lr_list.append(optimizer.param_groups[0]['lr'])
    
        print(f"Epoch {epoch+1}: Train Loss {train_loss[-1]:.4f}, Train Acc {train_acc[-1]:.4f}, Val Loss {val_loss_epoch:.4f}, Val Acc {val_acc[-1]:.4f}")
    
        if val_loss_epoch < best_loss:
            best_loss = val_loss_epoch
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved at epoch {epoch+1}")





plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()





plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Validation Acc')
plt.legend()
plt.show()





model.load_state_dict(torch.load("best_model.pth", {epoch+1}))
model.eval()





with torch.no_grad():
    running_test_loss, correct_test = [], 0
    for x, targets in test_loader:
        x, targets = x.to(device), targets.to(device)
        x = x.view(x.size(0), -1)
        preds = model(x)
        loss = loss_fn(preds, targets)
        running_test_loss.append(loss.item())
        correct_test += (preds.argmax(dim=1) == targets).sum().item()
    
    test_loss = np.mean(running_test_loss)
    test_acc = correct_test / len(test_data)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

