# 2023/12/21
# zhangzhong
# Pretrain BERT

import torch
from torch import nn, Tensor


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_len: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_len = max_len
    
        # 1. token embedding
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=hidden_size)
        # 2. segment embedding
        self.segment_embedding = nn.Embedding(num_embeddings=2,
                                              embedding_dim=hidden_size)
        # 3. positional embedding
        self.positional_embedding = nn.Parameter(data=torch.randn(size=(1, max_len, hidden_size)))
    
    def forward(self, x: tuple[Tensor, Tensor]):
        # result = token + segment + positional
        
        # token embedding
        source, segment = x
        batch_size, seq_size = source.shape
        y = self.token_embedding(source)
        assert y.shape == (batch_size, seq_size, self.hidden_size)
        
        # segment embedding
        assert segment.shape == (batch_size, seq_size)
        segment_embedding = self.segment_embedding(segment)
        assert segment_embedding.shape == (batch_size, seq_size, self.hidden_size)

        # positional embedding
        


class BERTEncoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
class NextSentencePrediction(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
# TODO: MaskedLanguageModel

class BERT(nn.Module):
    def __init__(self):
        # 1. Embedding
        self.embedding = BERTEmbedding()
        # 2. BERTEncoder
        # 3. NextSentencePrediction + MaskedLanguageModel
        pass
    
    def forward(self, source: Tensor):
        pass
