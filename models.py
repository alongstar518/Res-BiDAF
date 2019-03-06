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
    def __init__(self, word_vectors, hidden_size, max_p_len, max_q_len,drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        self.emb_trans = layers.EmbeddingTransformer(word_vectors=word_vectors,
                                                     drop_prob=drop_prob)


        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        self.enc_trans = layers.TransformerEncoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=6,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )
        self.dec_trans = layers.TransformerDecoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=6,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )


        #self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
        #                                 drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=4 * word_vectors.size(-1),
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        #self.mod = layers.RNNEncoder(input_size=4 * hidden_size,
        #                             hidden_size=hidden_size,
        #                             num_layers=2,
        #                             drop_prob=drop_prob)

        self.pe_p = layers.PositionalEncoder(word_vectors.size(-1), max_p_len)
        self.pe_q = layers.PositionalEncoder(word_vectors.size(-1), max_q_len)
        self.out = layers.BiDAFOutput(hidden_size=word_vectors.size(-1),q_length=max_q_len,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs #(batch, max p length)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs #(batch, max_q_length)
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        #c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        #q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_emb = self.emb_trans(cw_idxs) #(batch, max p length, embedding)
        q_emb = self.emb_trans(qw_idxs) #(batch, max q length, embedding)

        #c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        #q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)
        c_emb = self.pe_p(c_emb)  # same as c_enc
        q_emb = self.pe_q(q_emb)  # same as q_enc
        c_enc = self.enc_trans(c_emb,c_mask)  #same as c_emb
        q_enc = self.enc_trans(q_emb,q_mask) # same as q_emb
        dec_out = self.dec_trans(q_emb, c_enc, c_mask, q_mask) # same as q_enc

        #att = self.att(c_enc, q_enc,
                       #c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        #mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
        out = self.out(q_enc, dec_out,c_enc,c_mask)  # 2 tensors, each (batch_size, c_len)
                #, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
