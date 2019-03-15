"""Top-level model classes.

"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, char_vocab_size, max_p_length, max_q_length, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    char_vocab_size=char_vocab_size,
                                    char_embedding_size=100,
                                    word_embedding_size=300,
                                    kernel_size=5,
                                    drop_prob=drop_prob)

        self.enc_trans = layers.TransformerEncoder(input_size=4 *(word_vectors.size(-1) + 100),
                                                   num_q= 64,
                                                   num_k= 64,
                                                   num_v= 64,
                                                   num_head=8,
                                                   num_layer=6,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )

        self.enc = layers.RNNEncoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

        self.highway = layers.Highway(1, hidden_size)

        self.pe_p = layers.PositionalEncoder(4 * (word_vectors.size(-1)+100), max_seq_len=max_p_length)

        self.relu = nn.ReLU()

    def forward(self, cw_idxs, qw_idxs, cw_char_idx, qw_char_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cw_char_idx)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qw_char_idxs)         # (batch_size, q_len, hidden_size)

        att = self.att(c_emb, q_emb,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        att2 = self.pe_p(att)

        att2 = self.enc_trans(att2, c_mask)    # (batch_size, c_len, 2 * hidden_size)

        att = att + att2

        self.relu(att)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
