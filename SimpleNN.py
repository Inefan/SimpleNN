import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import os
import matplotlib.pyplot as plt
import numpy as np

import json
from tqdm import tqdm
from PIL import Image

from torchvision.transforms import v2


plt.style.use("dark_background")

device = "cuda" if torch.cuda.is_available() else "cpu"
device


class MNISTDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)  
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = os.path.basename(path_dir)  

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))

            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset
        
    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = Image.open(file_path) 

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=(0.5, ), std=(0.5, ))
    ]
)

train_data = MNISTDataset("./mnist/training", transform=transform)
test_data = MNISTDataset("./mnist/training", transform=transform)

train_data, val_data = random_split(train_data, [0.7, 0.3])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class MyModel(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer_1 = nn.Linear(input, 128)
        self.layer_2 = nn.Linear(128, output)
        self.act = nn.ReLU()  

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        out = self.layer_2(x)
        return out


model = MyModel(784, 10).to(device)

input = torch.rand([16, 784], dtype=torch.float32).to(device)

out = model(input)
out.shape


loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

EPOCHS = 1
train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
best_loss = None
count = 0

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = []
    true_answer = 0
    train_loop = tqdm(train_loader, leave=False)
    for x, targets in train_loop:
        x = x.reshape(-1, 28*28).to(device)
        targets = targets.reshape(-1).to(torch.int32)
        targets = torch.eye(10)[targets].to(device)

        pred = model(x)
        loss = loss_model(pred, targets)

        opt.zero_grad()
        loss.backward()

        opt.step()

        running_train_loss.append(loss.item())
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}], train_loss={mean_train_loss:.4f}")

    running_train_acc = true_answer / len(train_data)

    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)

    model.eval()
    with torch.no_grad():
        running_val_loss = []
        true_answer = 0
        for x, targets in val_loader:
            x = x.reshape(-1, 28*28).to(device)

            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(10)[targets].to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)

            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        running_val_acc = true_answer / len(val_data)

        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

        lr_scheduler.step(mean_val_loss)
        lr = lr_scheduler._last_lr[0] 
        lr_list.append(lr)

        print(f"Epoch [{epoch+1}/{EPOCHS}], train_loss={mean_train_loss:.4f}, train_acc={running_train_acc:.4f}, val_loss={mean_val_loss:.4f}, val_acc={running_val_acc:.4f}")

        if best_loss is None:
            best_loss = mean_val_loss

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            count = 0

            checkpoint = {
                "state_model": model.state_dict(),
                "state_opt": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
                "loss": {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss
                },
                'metric': {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                },
                'lr': lr_list,
                'epoch': {
                    'EPOCHS': EPOCHS,
                    'save_epoch': epoch
                }
            }

            torch.save(checkpoint, f'model_state_dict_epoch_{epoch+1}.pt')
            print(f'На епосі -{epoch+1}, збережена модель зі значеннями функції втрачень на валідації -{mean_val_loss:.4f}', end='\n\n')

        if count >= 0:
            print(f'\033[31m Навчання призупинено на {epoch + 1} епосі.\033[0m')
            break

        count += 1

plt.plot(lr_list)
plt.legend(['lr_list'])
plt.show()

plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['loss_train', 'loss_val'])
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['acc_train', 'acc_val'])
plt.show()


checkpoint = torch.load('model_state_dict_epoch_19.pt')
model.load_state_dict(checkpoint['state_model'])

model.eval()
with torch.no_grad():
    running_test_loss = []
    true_answer = 0
    for x, targets in test_loader:
        x = x.reshape(-1, 28*28).to(device)
        targets = targets.reshape(-1).to(torch.int32)
        targets = torch.eye(10)[targets].to(device)

        pred = model(x)
        loss = loss_model(pred, targets)

        running_test_loss.append(loss.item())
        mean_test_loss = sum(running_test_loss) / len(running_test_loss)

        true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

    running_test_acc = true_answer / len(test_data)

print(f'test_loss = {mean_test_loss:.4f}, test_acc = {running_test_acc:.4f}', end='\n\n')
