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
        self.enc_trans_s = layers.TransformerEncoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=3,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )
        self.dec_trans_s = layers.TransformerDecoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=3,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )

        self.enc_trans_h = layers.TransformerEncoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=3,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )

        self.dec_trans_h = layers.TransformerDecoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=6,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )

        self.enc_trans_e = layers.TransformerEncoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=3,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )

        self.dec_trans_e = layers.TransformerDecoder(input_size=word_vectors.size(-1),
                                                   num_k=64,
                                                   num_v=64,
                                                   num_head=8,
                                                   num_layer=3,
                                                   hidden_size=hidden_size,
                                                   dropoutrate=drop_prob
                                                   )


        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=4 * word_vectors.size(-1),
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.pe_p = layers.PositionalEncoder(word_vectors.size(-1), max_p_len)
        self.pe_q = layers.PositionalEncoder(word_vectors.size(-1), max_q_len)
        self.out = layers.BiDAFOutput(hidden_size=word_vectors.size(-1),q_length=max_q_len,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs #(batch, max p length)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs #(batch, max_q_length)

        c_emb = self.emb_trans(cw_idxs) #(batch, max p length, embedding)
        q_emb = self.emb_trans(qw_idxs) #(batch, max q length, embedding)

        c_emb = self.pe_p(c_emb)  # same as c_enc
        q_emb = self.pe_q(q_emb)  # same as q_enc
        #c_enc = self.enc_trans(c_emb,c_mask)  #same as c_emb

        s_enc, _ = self.enc_trans_s(q_emb,q_mask) # same as q_emb (b, q, e)
        s_dec, _, s_de_attn = self.dec_trans_s(c_emb, s_enc, q_mask, c_mask) #(b, c, e)

        #h_enc, _ = self.enc_trans_h(s_dec, c_mask) #(b, c, e)
        #h_dec, _, h_de_attn = self.dec_trans_e(q_emb, h_enc, c_mask, q_mask) #(b, q, e)

        #e_enc, _ = self.enc_trans_e(h_dec, q_mask) #(b, q, e)
        #e_dec, _, e_de_attn = self.dec_trans_e(c_emb, e_enc, q_mask, c_mask) #(b,c,e)

        e_enc, _ = self.enc_trans_e(q_emb, q_mask) #(b, q, e)
        e_dec, _, e_de_attn = self.dec_trans_e(s_dec, e_enc, q_mask, c_mask) #(b,c,e)

        out = self.out(s_dec,e_dec,q_emb,c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
