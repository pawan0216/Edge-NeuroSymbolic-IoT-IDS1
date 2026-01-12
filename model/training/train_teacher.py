import torch
import pandas as pd
from model.neural_model import NeuralIDS

X_train = torch.tensor(pd.read_csv("X_train.csv").values).float()
y_train = torch.tensor(pd.read_csv("y_train.csv").values).long()

model = NeuralIDS(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "teacher_model.pt")
