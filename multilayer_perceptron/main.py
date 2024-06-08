import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader


class ElemPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(ElemPerceptron, self).__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_size, out_features=output_size),
            nn.Sigmoid())

    def forward(self, x):
        return self.linear_layer_stack(x)


def fit(model, loss_function, optimizer, train_loader, test_loader, epochs, statistic=True):
    for epoch in range(epochs):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx}")
            optimizer.zero_grad()

            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()


def find_max_index(array1, array2):
    max_index = array1.index(max(array1))
    if max_index == array2.index(1, 0):
        return True
    else:
        return False


train_data = datasets.MNIST(
    root='./data',
    train=True,
    transform=ToTensor(),
    download=True,
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.MNIST(
    root='./data',
    train=False,
    transform=ToTensor(),
    download=True,
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

batch_size = 100
learning_rate = 0.03
epochs = 10
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

elem_perceptron = ElemPerceptron(784, 10)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(elem_perceptron.parameters(), lr=learning_rate)
fit(elem_perceptron, loss_function, optimizer, train_loader, test_loader, epochs, True)

test_loss = 0.0
accuracy = 0
for data, target in test_loader:
    for i in range(len(data)):
        if find_max_index(data[i].detach().numpy().tolist(), target[i].detach().numpy().tolist()):
            accuracy += 1
    output = elem_perceptron(data)
    test_loss += loss_function(output, target).item()
print(f"Test loss: {test_loss / len(test_loader.dataset):.4f}, Accuracy: {accuracy / len(test_loader.dataset)}")
