from transformers import BertTokenizer, BertModel
import torch
import torch
from torch import nn
import numpy as np
from model import Transformer 

# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# source = torch.rand((10, 32, 512))
# target = torch.rand((20, 32, 512))
# out = transformer_model(source, target)
source = torch.rand((10, 32, 512))
target = source
model = Transformer()
output = model(source, target)

print('Source: \n', source)
print('Target: \n', target)
print('Output: \n', output)