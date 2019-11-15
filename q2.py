import torch
import pandas as pd

class ReQU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0).pow(2)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input = 2*input*grad_input
        return grad_input


class Net(torch.nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(4, 100)
        self.non_linearity = ReQU.apply
        self.fc2 = torch.nn.Linear(100, 3)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


data = pd.read_csv('iris.data.csv', header=None)
X = data.loc[:, range(4)]
y = data.loc[:, 4]
y.loc[y == 'Iris-setosa'] = 0
y.loc[y == 'Iris-versicolor'] = 1
y.loc[y == 'Iris-virginica'] = 2

train_X = torch.autograd.Variable(torch.tensor(X.to_numpy()).float())
train_y = torch.autograd.Variable(torch.tensor(y.to_numpy()).long())
net = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
for epoch in range(50000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data.item())

