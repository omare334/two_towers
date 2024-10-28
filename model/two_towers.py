import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


ENCODING_DIM = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

from typing import TypeAlias

Token: TypeAlias = int
Sequence: TypeAlias = list[Token]
# type Token = int
# type Sequence = list[Token]


class TwoTowers(nn.Module):
    def __init__(self, vocab_size, token_embed_dims, encoded_dim, rnn_layer_num=1):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=token_embed_dims
        )

        # Linear are just placeholders
        # self.query_rnn = nn.RNN(
        #     input_size=token_embed_dims,
        #     hidden_size=encoded_dim,
        #     num_layers=rnn_layer_num,
        #     batch_first=True,
        # )
        # self.doc_rnn = nn.RNN(
        #     input_size=token_embed_dims,
        #     hidden_size=encoded_dim,
        #     num_layers=rnn_layer_num,
        #     batch_first=True,
        # )
        self.query_lstm = nn.LSTM(
            input_size=token_embed_dims,
            hidden_size=encoded_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.doc_lstm = nn.LSTM(
            input_size=token_embed_dims,
            hidden_size=encoded_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=self.dist_function_single, reduction="mean"
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
        # Shape [N]
        query_lengths = torch.tensor([len(seq) for seq in query_tkns], dtype=torch.long)
        pos_lengths = torch.tensor(
            [len(seq) for seq in positive_tkns], dtype=torch.long
        )
        neg_lengths = torch.tensor(
            [len(seq) for seq in negative_tkns], dtype=torch.long
        )

        query_tkns = [
            torch.tensor(q, dtype=torch.long, device=device) for q in query_tkns
        ]
        positive_tkns = [
            torch.tensor(p, dtype=torch.long, device=device) for p in positive_tkns
        ]
        negative_tkns = [
            torch.tensor(n, dtype=torch.long, device=device) for n in negative_tkns
        ]

        # Shape [N, Lmax] (for each)
        padded_query_tkns = pad_sequence(query_tkns, batch_first=True)
        padded_pos_tkns = pad_sequence(positive_tkns, batch_first=True)
        padded_neg_tkns = pad_sequence(negative_tkns, batch_first=True)

        # Shape [N, Lmax, E]
        padded_query_embeds = self.embed(padded_query_tkns)
        padded_pos_embeds = self.embed(padded_pos_tkns)
        padded_neg_embeds = self.embed(padded_neg_tkns)

        packed_padded_queries = pack_padded_sequence(
            padded_query_embeds, query_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_pos = pack_padded_sequence(
            padded_pos_embeds, pos_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_neg = pack_padded_sequence(
            padded_neg_embeds, neg_lengths, batch_first=True, enforce_sorted=False
        )

        bi_query_encodings: torch.Tensor
        bi_pos_encodings: torch.Tensor
        bi_neg_encodings: torch.Tensor
        # Shape [2, N, H]
        _, (bi_query_encodings, _) = self.query_lstm(packed_padded_queries)
        _, (bi_pos_encodings, _) = self.doc_lstm(packed_padded_pos)
        _, (bi_neg_encodings, _) = self.doc_lstm(packed_padded_neg)

        batch_len = bi_query_encodings.shape[1]

        # Convert to [N, 2*H]
        query_encodings = bi_query_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        pos_encodings = bi_pos_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        neg_encodings = bi_neg_encodings.permute(1, 0, 2).reshape(batch_len, -1)

        return self.triplet_loss(query_encodings, pos_encodings, neg_encodings)

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

        _, query_rnn_embed = self.query_lstm(query_embed)
        _, pos_rnn_embed = self.doc_lstm(pos_embed)
        _, neg_rnn_embed = self.doc_lstm(neg_embed)
        return self.triplet_loss(query_rnn_embed, pos_rnn_embed, neg_rnn_embed)


if __name__ == "__main__":
    model = TwoTowers(200, 10, 20)

    batch_num = 20

    que_list = [
        torch.randint(0, 200, [torch.randint(5, 15, ())]) for _ in range(batch_num)
    ]
    pos_list = [
        torch.randint(0, 200, [torch.randint(5, 15, ())]) for _ in range(batch_num)
    ]
    neg_list = [
        torch.randint(0, 200, [torch.randint(5, 15, ())]) for _ in range(batch_num)
    ]

    loss = model.get_loss_batch(
        query_tkns=que_list, positive_tkns=pos_list, negative_tkns=neg_list
    )
    pass
