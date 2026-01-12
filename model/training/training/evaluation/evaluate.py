import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from model.neural_model import NeuralIDS
from model.fusion_model import neuro_symbolic_decision

X_test = torch.tensor(pd.read_csv("X_test.csv").values).float()
y_test = pd.read_csv("y_test.csv").values

model = NeuralIDS(X_test.shape[1])
model.load_state_dict(torch.load("student_model.pt"))
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probs, rules = neuro_symbolic_decision(logits, X_test)
    preds = probs.argmax(dim=1)

print("Accuracy:", accuracy_score(y_test, preds))
