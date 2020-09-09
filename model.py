import torch.nn as nn

class Transformer(nn.Module):
    # def __init__(self, num_emb=2, embed_dim=10):
    def __init__(self):
        super().__init__()

        # self.embed_src = nn.Embedding(num_embeddings=num_emb, embedding_dim=embed_dim)
        # self.embed_target = nn.Embedding(num_embeddings=num_emb, embedding_dim=embed_dim)
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        # self.linear = nn.Linear(10, 2)
        # self.softmax = nn.Softmax(dim = -1)

    def forward(self, source, target):
        # embed_src = self.embed_src(source)
        # embed_src = embed_src.view(len(source), 1, -1)

        # embed_target = self.embed_target(target)
        # embed_target = embed_target.view((len(target)), 1, -1)

        output = self.transformer(source, target)
        # output = self.linear(output)

        # return output.permute(0, 2, 1)
        return output