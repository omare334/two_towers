import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from w2v_model import Word2Vec

# from typing import TypeAlias

type Token = int
type Sequence = list[Token]


class TwoTowers(nn.Module):
    def __init__(self, vocab_size, token_embed_dims, encoded_dim, rnn_layer_num=1):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=token_embed_dims
        )

        # Linear are just placeholders
        self.query_rnn = nn.RNN(
            input_size=token_embed_dims,
            hidden_size=encoded_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
        )
        self.doc_rnn = nn.RNN(
            input_size=token_embed_dims,
            hidden_size=encoded_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
        )
        self.triplet_loss_single = nn.TripletMarginWithDistanceLoss(
            distance_function=self.dist_function_single
        )

    def dist_function_single(self, query, sample):
        return 1 - nn.functional.cosine_similarity(query, sample)

    def shape_batch(self, query_tkns, positive_tkns: torch.Tensor, negative_tkns):
        b, r, L = positive_tkns.shape
        pass

    def get_loss_batch(
        self,
        query_tkns: list[Sequence],
        positive_tkns: list[Sequence],
        negative_tkns: list[Sequence],
    ):
        # Expected shapes:
        # L - length of query/doc in tokens
        # b - number of batches
        # s - samples per query (both pos and neg)
        # query_tkns [b, L]
        query_lengths = [len(seq) for seq in query_tkns]
        padded_query_tkns = pad_sequence(query_tkns, True)

        pos_lengths = [len(seq) for seq in positive_tkns]
        padded_pos_tkns = pad_sequence(positive_tkns, True)

        neg_lengths = [len(seq) for seq in negative_tkns]
        padded_neg_tkns = pad_sequence(negative_tkns, True)

        padded_query_embeds = self.embed(padded_query_tkns)
        padded_pos_embeds = self.embed(padded_pos_tkns)
        padded_neg_embeds = self.embed(padded_neg_tkns)

        packed_query_embeds = pack_padded_sequence(
            padded_query_embeds, query_lengths, batch_first=True, enforce_sorted=False
        )
        packed_pos_embeds = pack_padded_sequence(
            padded_pos_embeds, pos_lengths, batch_first=True, enforce_sorted=False
        )
        packed_neg_embeds = pack_padded_sequence(
            padded_neg_embeds, neg_lengths, batch_first=True, enforce_sorted=False
        )

    def get_loss_single(
        self,
        query_tkns: list[Token],
        pos_tkns: list[Token],
        neg_tkns: list[Token],
    ):
        query_tkns = torch.LongTensor(query_tkns)
        pos_tkns = torch.LongTensor(pos_tkns)
        neg_tkns = torch.LongTensor(neg_tkns)

        query_embed = self.embed(query_tkns)
        pos_embed = self.embed(pos_tkns)
        neg_embed = self.embed(neg_tkns)

        _, query_rnn_embed = self.query_rnn(query_embed)
        _, pos_rnn_embed = self.doc_rnn(pos_embed)
        _, neg_rnn_embed = self.doc_rnn(neg_embed)
        return self.triplet_loss_single(query_rnn_embed, pos_rnn_embed, neg_rnn_embed)
