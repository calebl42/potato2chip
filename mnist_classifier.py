import io

import torch
import torch.nn as nn  # all neural network modules
import torch.optim as optim  # optimization algo
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, QConfig, \
    default_per_channel_weight_fake_quant
from torch.utils.data import DataLoader  # easier dataset management, helps create mini batches
import torchvision.datasets as datasets  # standard datasets
import torchvision.transforms as transforms
from torchinfo import summary

running_eval = False

class NN(nn.Module):  # inherits nn.Module

    def __init__(self, input_size, num_classes):  # input size = 28x28 = 784 for mnist
        super(NN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.fc1 = nn.Linear(input_size, 20, bias=False)
        self.activation = torch.nn.ReLU()
        self.fc2 = nn.Linear(20, num_classes, bias=False)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        #if not running_eval:
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    
batch_size = 64

# Load Data
train_dataset = datasets.MNIST(root='../dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Set Device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
num_epochs = 5

# Initialize Network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

model.eval()
model.qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=7,
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False
                ),
                weight=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=-4,
                    quant_max=3,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False
                )
            )
model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['fc1', 'activation']])
model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fp32_prepared.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):  
# 1 epoch => Network has seen all the images in the dataset
    
    print(f'Epoch: {epoch}')

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)  # => [64 , 1, 28, 28] => 64 : num_images, 1 -> num_channels, (28,28): (height, width)
        data = data.reshape(data.shape[0], -1)  # Flatten

        # forward
        scores = model_fp32_prepared(data)

        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()  # set all gradients to zero for each batch
        loss.backward()

        # gradient descent
        optimizer.step()


model_fp32_prepared.eval()
model_int4 = torch.ao.quantization.convert(model_fp32_prepared)


running_eval = True
torch.set_printoptions(profile="full", sci_mode=False, precision=2)

def check_accuracy(loader, model):

    if loader.dataset.train:
        print("Accuracy on training data")
    else:
        print("Accuracy on testing data")

    num_correct = 0
    num_samples = 0

    with torch.no_grad():  # dont compute gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)

            #print(scores.int_repr())

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100: .2f}')

    model.train()

# model_scripted = torch.jit.script(model_int4) # Export to TorchScript
# model_scripted.save('model_scripted.pt') # Save

# for w in model_fp32_prepared.parameters():
#     print(w)
#

# print(model_int4.quant)
# print(model_int4.quant.dtype)

torch.save(model_int4.state_dict(), 'state_dict')

# check_accuracy(train_loader, model)
# check_accuracy(train_loader, model_fp32_prepared)
check_accuracy(train_loader, model_int4)
check_accuracy(test_loader, model_int4)

