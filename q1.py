from __future__ import print_function
import torch


EPOCHS_TO_TRAIN = 15000

# Tensor for the truth table
x = torch.Tensor([[1, 1, 0, 0], [1, 0, 1, 0]])
y = torch.Tensor([[0, 1, 1, 0]])
D_in, H, D_out = 2, 3,  1

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):
    for t in range(4):
        y_pred = model(x[:, t])
        loss = loss_fn(y_pred, y[:, t])
        optimizer.zero_grad()   # zero the gradient buffers
        loss.backward()
        optimizer.step()    # Does the update
    if idx % 1000 == 0:
        print("Epoch {: >8} Loss: {}".format(idx, loss.data.item()))

print("")
print("Final results:")
for t in range(4):
    inp = x[:, t]
    output = model(x[:, t])
    target = y[:, t]
    print("Input:[{}, {}] Target:[{}] Predicted:[{}]".format(
        int(inp.data[0].item()),
        int(inp.data[1].item()),
        int(target.data.item()),
        output.data.item()
            ))
