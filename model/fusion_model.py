import torch.nn.functional as F
from model.symbolic_rules import symbolic_rules

def neuro_symbolic_decision(logits, features):
    probs = F.softmax(logits, dim=1)
    rules = symbolic_rules(features)

    if len(rules) > 0:
        probs[:,1] += 0.05  # symbolic bias

    return probs, rules
