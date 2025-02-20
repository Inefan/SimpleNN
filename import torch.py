import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32) 
y_train = 2 * (x_train ** 2) - x_train + 5 
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(1, 20)  
        self.relu = nn.ReLU()           
        self.linear2 = nn.Linear(20, 1)  

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)  
        return self.linear2(x)  


model = SimpleNN()
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01) 






epochs = 200  
losses = []  
for epoch in range(epochs):
    y_pred = model(x_train)  
    loss = criterion(y_pred, y_train)  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    losses.append(loss.item())  
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()





x_test = torch.tensor([[5.0], [6.0], [7.0]], dtype=torch.float32)  
y_test_pred = model(x_test) 
y_actual = 2 * (x_test ** 2) - x_test + 5


print("Test predictions:")
print(y_test_pred)
print("Expected results:")
print(y_actual)







