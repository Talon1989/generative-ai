import torch
import torch.nn as nn
import string
import random
import sys
import unidecode

'''
based on
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Projects/text_generation_babynames/generating_names.py
'''

# Example model output and target labels
model_output = torch.tensor([[1.2, -0.5, 0.8, 0.5], [0.9, 2.1, -1.0, 0.5], [-1.5, 0.6, 2.0, 3.]])
target_labels = torch.tensor([0, 1, 2])  # Sparse labels

print(model_output.shape)
print(target_labels.shape)

# Create the loss function
loss_function = nn.CrossEntropyLoss()

# Calculate the loss
loss = loss_function(model_output, target_labels)

print("Cross-Entropy Loss:", loss.item())
























































































































































