from transformers import BertTokenizer, BertModel
import torch
import torch
from torch import nn
import numpy as np
from model import Transformer 

source = torch.rand((10, 1, 512))
target = source

def train(model, num_epochs=10, learning_rate=0.01):
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(num_epochs):
        prediction = model(source, target)

        if i == num_epochs-1:
            print('Prediction: ', prediction)

        loss = loss_function(prediction, target)
        print('Loss: ', loss.item())

        optimizer.zero_grad()
        loss.backward
        optimizer.step()

model = Transformer()
train(model)