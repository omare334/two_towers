import torch
import torch.nn as nn
from rnn import RNN_model
from w2v_model import Word2Vec
from typing import TypeAlias
from typing import List

Token = int

class TwoTowers(nn.Module):
    def __init__(self, vocab_size, token_embed_dims, query_embed_dim, rnn_layer_num=2):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=token_embed_dims
        )

        # Linear are just placeholders
        self.query_rnn = nn.RNN(
            input_size=token_embed_dims,
            hidden_size=query_embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
        )
        self.doc_rnn = nn.RNN(
            input_size=token_embed_dims,
            hidden_size=query_embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
        )
        self.triplet_loss_single = nn.TripletMarginWithDistanceLoss(
            distance_function=self.dist_function_single
        )

    def dist_function_single(self, query, sample):
        return 1 - nn.functional.cosine_similarity(query, sample)

    # def shape_batch(self, query_tkns, positive_tkns: torch.Tensor, negative_tkns):
    #     b, r, L = positive_tkns.shape
    #     pass

    # def get_loss_batch(
    #     self,
    #     query_tkns: List[List[int]],
    #     positive_tkns: List[List[List[int]]],
    #     negative_tkns: List[List[List[int]]],
    # ):
    #     # Expected shapes:
    #     # L - length of query/doc in tokens
    #     # b - number of batches
    #     # s - samples per query (both pos and neg)
    #     # query_tkns [b, L]
    #     pass

    def get_loss_single(
        self,
        query_tkns: List[Token],
        pos_tkns: List[Token],
        neg_tkns: List[Token],
    ):
        query_tkns = torch.LongTensor(query_tkns).unsqueeze(0)
        pos_tkns = torch.LongTensor(pos_tkns).unsqueeze(0)
        neg_tkns = torch.LongTensor(neg_tkns).unsqueeze(0)

        query_embed = self.embed(query_tkns)
        pos_embed = self.embed(pos_tkns)
        neg_embed = self.embed(neg_tkns)

        _, query_rnn_embed = self.query_rnn(query_embed)
        _, pos_rnn_embed = self.doc_rnn(pos_embed)
        _, neg_rnn_embed = self.doc_rnn(neg_embed)

        return self.triplet_loss_single(query_rnn_embed, pos_rnn_embed, neg_rnn_embed)