"""Assortment of layers for use in models.py.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import math
import numpy as np

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class EmbeddingTransformer(nn.Module):
    """Embedding layer used by BiDAF Transfomer encoder, without the character-level component.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, drop_prob = 0.1):
        super(EmbeddingTransformer, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        #emb = F.dropout(emb, self.drop_prob, self.training)

        return emb

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=0):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False).to(x.device)
        return x


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def  __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x

class TransformerEncoderCell(nn.Module):
    def __init__(self, input_size, num_k, num_v, num_head, hidden_size, dropoutrate = 0.1):
        super(TransformerEncoderCell, self).__init__()
        self.self_attn = MultiHeadAttention(num_head, input_size, num_k, num_v, dropout=dropoutrate)
        self.feed_forward = FeedForward(input_size, hidden_size, input_size)
        #self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, q,k,v, mask, attention_mask):
        z, attn = self.self_attn(q,k,v, attention_mask)
        z *= mask
        #z = self.layer_norm(x + z)
        x = self.feed_forward(z)
        #x = self.layer_norm(x + z)
        x *= mask
        return x, attn

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_k, num_v, num_head, hidden_size, num_layer=6, dropoutrate = 0.1):
        super(TransformerEncoder, self).__init__()
        self.transformer_cells = nn.ModuleList([TransformerEncoderCell(input_size, num_k, num_v, num_head, hidden_size, dropoutrate)
                                  for _ in range(num_layer)])

        #self.projection = nn.Linear(hidden_size, 2*hidden_size, bias=False)

    def forward(self, x, mask):
        attn = []
        l_q = x.size(1)
        attn_mask = (1-mask).unsqueeze(1).expand(-1, l_q, -1)
        mask = mask.float().unsqueeze(-1).expand(-1, -1, x.size(-1))
        for cell in self.transformer_cells:
            x, att = cell(x,x,x, mask,attn_mask)
            attn += [att]
        return x, attn

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, num_k, num_v, num_head, hidden_size, num_layer=6, dropoutrate = 0.1):
        super(TransformerDecoder, self).__init__()
        self.transformer_cells = nn.ModuleList([DecoderCell(input_size, num_k, num_v, num_head, hidden_size, dropoutrate)
                                  for _ in range(num_layer)])

    def forward(self, dec_input, enc_output, p_mask, q_mask):
        padding_mask = q_mask.type(torch.float).unsqueeze(-1)
        l_q = dec_input.size(1)
        sz_b, len_s, _ = dec_input.size()
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=p_mask.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
        attn_mask = (1-q_mask).unsqueeze(1).expand(-1, l_q, -1)
        att_mask = (attn_mask + subsequent_mask).gt(0)
        l_q = dec_input.size(1)
        dec_enc_attn_mask = (1 - p_mask).unsqueeze(1).expand(-1, l_q, -1)
        attn = []
        de_attn = []
        for cell in self.transformer_cells:
            x,att,de_att = cell(dec_input,enc_output, padding_mask,att_mask, dec_enc_attn_mask)
            attn += [att]
            de_attn += [de_att]
        return x, attn, de_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class SelfAttention(nn.Module):
    def __init__(self, input_size,num_k, num_v,num_head, dropoutrate):
        '''
        SelfAttention layer initilization.
        :param input_size: scalar,  embedding size.
        :param out_size: scalar, final output size
        :param num_k: scalar, k size
        :param num_v:  scalar, v size
        :param dropout: dropout rate, scalar
        '''
        super(SelfAttention,self).__init__()
        self.num_k = num_k
        self.num_v = num_v
        self.input_size = input_size
        self.num_head = num_head

        self.linear_q = nn.Linear(input_size, num_head * num_k, bias=False)
        self.linear_k = nn. Linear(input_size, num_head * num_k, bias=False)
        self.linear_v = nn.Linear(input_size, num_head * num_v, bias=False)
        self.temperature = math.sqrt(num_k)
        self.softmax = nn.Softmax(dim = 2)
        self.fc = nn.Linear(num_head * num_v,input_size, bias=False)
        self.dropout = nn.Dropout(p=dropoutrate)

    def forward(self, q, k, v, softmax_mask):
        '''
        forward for self attention
        :param x: input embeddings (batch, passage_length, embeddingsize)
        :return: attn: (batch, passage_length, embeddingsize)
        '''
        q = self.linear_q(q) # (batch, passage_length, num_head * num_k)
        k = self.linear_k(k) # (batch, passage_length, num_head * num_k)
        v = self.linear_v(v) # (batch, passage_length, num_head * num_v)

        x = torch.bmm(q,k.transpose(1,2)) / self.temperature
        x = x.data.masked_fill_(softmax_mask.byte(), -float('inf'))
        x = self.softmax(x)
        x = self.dropout(x)
        x = torch.bmm(x, v)
        attn = x
        x = self.fc(x)
        return x, attn

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class DecoderCell(nn.Module):
    def __init__(self, input_size, num_k, num_v, num_head, hidden_size, dropoutrate = 0.1):
        super(DecoderCell, self).__init__()
        self.self_attn = MultiHeadAttention(num_head, input_size, num_k, num_v, dropout=dropoutrate)
        self.encoder_attn = MultiHeadAttention(num_head, input_size, num_k, num_v, dropout=dropoutrate)
        self.feed_forward = FeedForward(input_size, hidden_size, input_size)

    def forward(self, dec_input, enc_output, mask, self_att_mask, dec_enc_mask):
        dec_out, dec_input_attn = self.self_attn(dec_input, dec_input, dec_input, self_att_mask)
        dec_out *= mask

        dec_out, dec_enc_attn = self.encoder_attn(dec_out, enc_output, enc_output, dec_enc_mask)
        dec_out *= mask

        dec_out = self.feed_forward(dec_out)
        dec_out *= mask

        return dec_out, dec_input_attn, dec_enc_attn



class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        #self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        #self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.c_weight = nn.Parameter(torch.zeros(300, 1))
        self.q_weight = nn.Parameter(torch.zeros(300, 1))
        #self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, 300))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class BiLinearAttention(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(BiLinearAttention, self).__init__()
        self.linear = nn.Linear(input_feature, output_feature)

    def forward(self, p, q):
        '''

        :param p: batch, sentense_lengh_p, embedding
        :param q: batch, sentense_lengh_q, embedding
        :return: batch, sentense_length_p, sentense_length_q)
        '''
        attn = torch.bmm(F.relu(self.linear(p)),q.transpose(1,2))  #(batch, sentense_length_p, sentense_length_q)

        return attn

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, q_length, drop_prob):
        super(BiDAFOutput, self).__init__()

        self.attn_s = BiLinearAttention(hidden_size,hidden_size)
        self.attn_e = BiLinearAttention(hidden_size,hidden_size)
        self.fc_s = nn.Linear(hidden_size,1)
        self.fc_e = nn.Linear(hidden_size,1)

    def forward(self, dec_start, dec_end, enc_s, p_mask):
        # Shapes: (batch_size, seq_len, 1)
        #att1 = self.attn_s(dec_s,enc_s)
        #att2 = self.attn_e(dec_s,enc_s)
        #dec_s = torch.split((dec_s, p_mask.size(0)))
        #dec_s = torch.sum(dec_s, 0)
        #dec_e = torch.split((dec_e, p_mask.size(0)))
        #dec_e = torch.sum(dec_e, 0)
        torch.set_printoptions(profile='full')
        #dec_s = torch.split(dec_start[0], enc_s.size(0), 0)
        #dec_s = torch.stack(dec_s)
        #dec_s = torch.max(dec_s, 0)[0]

        #dec_e = torch.split(dec_end[0], enc_s.size(0), 0)
        #dec_e = torch.stack(dec_e)
        #dec_e = torch.max(dec_e, 0)[0]
        lg1 = self.fc_s(dec_start)
        lg2 = self.fc_e(dec_end)
        #lg1 = dec_s.sum(dec_s, -1)
        #lg2 = dec_e.sum(dec_e, -1)
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(lg1.squeeze(), p_mask, log_softmax=True)
        log_p2 = masked_softmax(lg2.squeeze(), p_mask, log_softmax=True)

        return log_p1, log_p2