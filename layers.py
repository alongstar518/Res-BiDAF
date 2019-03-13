"""Assortment of layers for use in models.py.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import math

class CharEmbedding(nn.Module):

    def __init__(self, char_vocab_size,char_embedding_size, word_embedding_size,kernel_size=5):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_embedding_size, padding_idx=0)
        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size
        self.cov1d_layer = nn.Conv1d(in_channels=self.char_embedding_size, out_channels=self.char_embedding_size, kernel_size=kernel_size, bias=True)
        #self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        """

        :param input: (Batch_size , Sentense_lenth, max_char_length)
        :return: word embedding vectors for sentense. (Batches_size*sentense_length, word_embedding_size)
        """
        batch_size = x.size(0)
        max_word_lenth = x.size(2)
        max_sent_lenth = x.size(1)
        x = x.contiguous().view(max_sent_lenth*batch_size, max_word_lenth)
        x = self.embedding(x)
        #x = self.drop_out(x)
        x = x.permute(0,2,1)
        conv = self.cov1d_layer(x)
        relu = F.relu(conv)
        out = torch.max(relu, dim=2)[0]
        out = out.view(batch_size, max_sent_lenth, self.char_embedding_size)
        return out


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size,char_vocab_size,char_embedding_size, word_embedding_size, kernel_size,drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)
        #self.char_embedding = CharEmbedding(char_vocab_size, char_embedding_size, channel_size, channel_width,drop_prob)
        self.char_embedding = CharEmbedding(char_vocab_size,char_embedding_size, char_embedding_size, kernel_size)
        self.drop_out = nn.Dropout(self.drop_prob)

    def forward(self, w, c):
        emb_c = self.char_embedding(c)
        emb_w = self.embed(w)   # (batch_size, seq_len, embed_size)
        emb = self.hwy(emb_c, emb_w)   # (batch_size, seq_len, hidden_size)
        emb = self.drop_out(emb)
        return emb


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
        self.transforms = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size * 2)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size * 2)
                                    for _ in range(num_layers)])

    def forward(self, x_c, x_w):
        x = torch.cat([x_c, x_w], dim=-1)
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x

class TransformerEncoderCell(nn.Module):
    def __init__(self, input_size, num_k, num_v, num_head, hidden_size, dropoutrate = 0.1):
        super(TransformerEncoderCell, self).__init__()
        self.self_attn = SelfAttention(input_size, num_k, num_v, num_head, dropoutrate)
        self.feed_forward = FeedForward(input_size, hidden_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, softmax_mask):
        z = self.self_attn(x, softmax_mask)
        z = self.layer_norm(x + z)
        x = self.feed_forward(z)
        x = self.layer_norm(x + z)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
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
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], \
                         requires_grad=False).to(x.device)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_k, num_v, num_head, hidden_size, num_layer=6, dropoutrate = 0.1):
        super(TransformerEncoder, self).__init__()
        self.transformer_cells = nn.ModuleList([TransformerEncoderCell(input_size, num_k, num_v, num_head, hidden_size, dropoutrate)
                                  for _ in range(num_layer)])

    def forward(self, x, mask):
        l_q = x.size(1)
        if mask is not None:
            attn_mask = (1-mask).unsqueeze(1).expand(-1, l_q, -1)
        else:
            attn_mask = None
        for cell in self.transformer_cells:
            x = cell(x, attn_mask)
        if mask is not None:
            mask = mask.float().unsqueeze(-1).expand(-1, -1, x.size(-1))
            x *= mask
        return x

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

    def forward(self, x, softmax_mask):
        '''
        forward for self attention
        :param x: input embeddings (batch, passage_length, embeddingsize)
        :return: attn: (batch, passage_length, embeddingsize)
        '''
        q = self.linear_q(x) # (batch, passage_length, num_head * num_k)
        k = self.linear_k(x) # (batch, passage_length, num_head * num_k)
        v = self.linear_v(x) # (batch, passage_length, num_head * num_v)

        x = torch.bmm(q,k.transpose(1,2)) / self.temperature
        if softmax_mask is not None:
            x = x.data.masked_fill_(softmax_mask.byte(), -float('inf'))
        x = self.softmax(x)
        x = self.dropout(x)
        x = torch.bmm(x, v)
        x = self.fc(x)
        return x

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
    def __init__(self,
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
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
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
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
