import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Multi_Head_Attention(nn.Module):
    def __init__(self, d_v, d_k, d_q, d_model, h, N_seq_1, dropout_dot_product = 0.1, dropout_fc = 0.1, N_seq_2 = None):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_model = d_model
        self.N = N_seq_1

        # if N_seq_2 is not none ==> Multi Head Attention for joining encoder and decoder sequence. In this case, N_seq_2 represents the encoder seq. length
        if N_seq_2 is not None:
            self.N_2 = N_seq_2

        else:
            self.N_2 = self.N

        self.k_norm = np.sqrt(d_k)
        self.fc = nn.Linear(h * d_v, d_model)

        # TODO REF https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/20f355eb655bad40195ae302b9d8036716be9a23/transformer/SubLayers.py#L19
        self.linear_v = nn.Linear(d_model, h * d_v)
        self.linear_k = nn.Linear(d_model, h * d_k)
        self.linear_q = nn.Linear(d_model, h * d_q)

        self.dropout_dot_product = nn.Dropout(dropout_dot_product)
        self.dropout_fc = nn.Dropout(dropout_fc)

        # attention matrices
        self.attention = {}

        self.collect_attention_maps = False
        self.i_iteration = -1

    def return_attention_maps(self, state):
        self.collect_attention_maps = state

    def set_iteration(self, i_iteration):
        self.i_iteration = i_iteration

    def forward(self, v, k, q):
        self.batch_size = v.shape[0]

        assert (self.batch_size, self.N_2, self.d_model) == v.shape
        assert (self.batch_size, self.N_2, self.d_model) == k.shape
        assert (self.batch_size, self.N, self.d_model) == q.shape

        v = self.linear_v(v)
        k = self.linear_k(k)
        q = self.linear_q(q)

        assert (self.batch_size, self.N_2, self.d_v * self.h) == v.shape
        assert (self.batch_size, self.N_2, self.d_k * self.h) == k.shape
        assert (self.batch_size, self.N, self.d_k * self.h) == q.shape

        # v * attention(q, k)

        # if multihead
        v, k, q = self.serialise(v, k, q)

        assert (self.batch_size * self.h, self.N_2, self.d_v) == v.shape
        assert (self.batch_size * self.h, self.N_2, self.d_k) == k.shape
        assert (self.batch_size * self.h, self.N, self.d_k) == q.shape

        v_att = self.scaled_dot_product(v, k, q)

        # if multihead
        assert (self.batch_size * self.h, self.N, self.d_v) == v_att.shape

        v_att = v_att.reshape(self.h, self.batch_size, self.N, self.d_v)
        v_att = v_att.permute(1, 2, 0, 3)
        v_att = v_att.reshape(self.batch_size, self.N, self.d_v * self.h)

        assert (self.batch_size, self.N, self.d_v * self.h) == v_att.shape

        v_att = self.fc(v_att)
        v_att = self.dropout_fc(v_att)

        assert (self.batch_size, self.N, self.d_model) == v_att.shape

        return v_att

    def scaled_dot_product(self, v, k, q):
        """
         v = (B, N, d)
         k = (B, N, d)
         q = (B, N, d)

         d = d_q = d_k

         Q * K^T = (B, N, d) * (B, d, N) = (B, N, N)
         Q * K^T * V = (B, N, N) * (B, N, d_k) = (B, N, d_k)

        :param v:
        :param k:
        :param q:
        :return:
        """

        assert q.shape[-1] == k.shape[-1]
        batch_size = v.shape[0]

        # multihead
        assert self.batch_size * self.h == batch_size

        att = torch.bmm(q, k.transpose(1, 2))

        assert (batch_size, self.N, self.N_2) == att.shape

        att /= self.k_norm
        att = F.softmax(att, dim = 2)
        att = self.dropout_dot_product(att)
        v_att = torch.bmm(att, v)
        assert (batch_size, self.N, self.d_v) == v_att.shape

        if self.collect_attention_maps:
            assert self.i_iteration > -1

            # reset
            if self.i_iteration == 0:
                self.attention = {}

            self.attention[str(self.i_iteration)] = att.reshape(self.h, self.batch_size, self.N, self.N_2).transpose(0,
                                                                                                                     1).squeeze(
                0)
        # print("multi head {}".format(self.i_iteration))

        return v_att

    def serialise(self, v, k, q):
        # TODO REF https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/20f355eb655bad40195ae302b9d8036716be9a23/transformer/SubLayers.py#L45

        v = v.reshape(self.batch_size, self.N_2, self.h, self.d_v)
        k = k.reshape(self.batch_size, self.N_2, self.h, self.d_k)
        q = q.reshape(self.batch_size, self.N, self.h, self.d_k)

        v = v.permute(2, 0, 1, 3)
        k = k.permute(2, 0, 1, 3)
        q = q.permute(2, 0, 1, 3)

        v = v.reshape(self.batch_size * self.h, self.N_2, self.d_v)
        k = k.reshape(self.batch_size * self.h, self.N_2, self.d_k)
        q = q.reshape(self.batch_size * self.h, self.N, self.d_k)

        return v, k, q


class Position_FF(nn.Module):
    def __init__(self, d_model, d_hidden_position_ff, kernel_size):
        super().__init__()

        self.position_ff = nn.Sequential(*[
            nn.Conv1d(d_model, d_hidden_position_ff, kernel_size),
            nn.ReLU(),
            nn.Conv1d(d_hidden_position_ff, d_model, kernel_size)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.position_ff(x)
        x = x.transpose(1, 2)

        return x


class Encoder_Layer(nn.Module):
    def __init__(self, d_v, d_k, d_q, d_model, h, d_hidden_position_ff, N, dropout_dot_product = 0.1, dropout_fc = 0.1,
                 dropout_pos_ff = 0.1, old_transformer_model = False):
        super().__init__()

        self.old_transformer_model = old_transformer_model
        self.d_model = d_model
        self.N = N

        self.multi_head_attention = Multi_Head_Attention(d_v, d_k, d_q, d_model, h, N,
                                                         dropout_dot_product = dropout_dot_product,
                                                         dropout_fc = dropout_fc)
        self.dropout_pos_ff = nn.Dropout(dropout_pos_ff)

        kernel_size = 1
        # TODO commenting out class implementation of Position_FF only because faster restoring of checkpoints from outdated model structure architectures

        if self.old_transformer_model:
            self.position_ff = nn.Sequential(*[
                nn.Conv1d(d_model, d_hidden_position_ff, kernel_size),
                nn.ReLU(),
                nn.Conv1d(d_hidden_position_ff, d_model, kernel_size)
            ])

        else:
            self.position_ff = Position_FF(d_model, d_hidden_position_ff, kernel_size)

        self.layer_norm_attention = nn.LayerNorm(self.d_model)
        self.layer_norm_FF = nn.LayerNorm(self.d_model)

    # def add_and_norm(self, x, residual_conn, dim):
    # add a residual connection residual_conn to x and normalise to shape n
    #	x = F.layer_norm(x + residual_conn, (dim,))
    #	return x

    def return_attention_maps(self, state):
        self.multi_head_attention.return_attention_maps(state)

    def set_iteration(self, i_iteration):
        self.multi_head_attention.set_iteration(i_iteration)

    def forward(self, sequence):
        self.batch_size = sequence.shape[0]
        assert (self.batch_size, self.N, self.d_model) == sequence.shape

        v_att = self.multi_head_attention(sequence, sequence, sequence)

        assert (self.batch_size, self.N, self.d_model) == v_att.shape

        # add residual and normalise
        # v_enc_norm = self.add_and_norm(v_att, sequence, self.d_model)
        v_enc_norm = self.layer_norm_attention(v_att + sequence)

        assert (self.batch_size, self.N, self.d_model) == v_enc_norm.shape

        # Positional encoding: (B, d_model, N) ==> CNN (Batch, Channels, L)

        # TODO only use transpose if sequence is used for sake of faster checkpoint loading of older models
        if self.old_transformer_model:
            v_pos = self.position_ff(v_enc_norm.transpose(1, 2)).transpose(1, 2)

        else:
            v_pos = self.position_ff(v_enc_norm)

        assert (self.batch_size, self.N, self.d_model) == v_pos.shape

        v_pos = self.dropout_pos_ff(v_pos)

        # add residual
        # encoder_output = self.add_and_norm(v_pos, v_enc_norm, self.d_model)
        encoder_output = self.layer_norm_FF(v_pos + v_enc_norm)
        assert (self.batch_size, self.N, self.d_model) == encoder_output.shape

        return encoder_output


class Encoder(nn.Module):
    def __init__(self, N, d_model = None, n_layers = None, h = None, d_ff = None, d_v = None, d_k = None, d_q = None,
                 dropout_dot_product = 0.1, dropout_fc = 0.1, dropout_pos_ff = 0.1, old_transformer_model = False):
        """
        paper default values

        d_model = 512
        n_layers = 6
        h = 8
        d_v = d_k = d_model/h
        d_ff = 2048

        Encoder module of the Transformer.
        dropout vals by radford2018improving


        :param n_layers: Number of encoder layers
        :param h: number of multi head attention maps
        :param d_model: dimension of every sequence's element
        :param N: number of sequence elements
        :param d_ff: dimension of position aware feed forward network
        """
        super().__init__()

        assert (d_v is not None and d_k is not None and d_q is not None) or (
                    d_v is None and d_k is None and d_q is None)

        if d_model is None:
            # default: paper implementation
            self.d_model = 512

        else:
            self.d_model = d_model

        if n_layers is None:
            # default: paper implementation
            n_layers = 6

        if h is None:
            # default: paper h = 8
            h = 8

        if d_v is None and d_k is None and d_q is None:
            # default: paper implementation d_k = d_v = d_model/h
            d_v = int(d_model / h)
            d_q = int(d_model / h)
            d_k = int(d_model / h)

        # d_v, d_k, d_q, d_model = d, d, d, d

        self.N = N

        if d_ff is None:
            # paper: d = 512, d_ff = 2048
            d_ff = 2048

        self.encoder_layers = nn.Sequential(*[
            Encoder_Layer(d_v, d_k, d_q, d_model, h, d_ff, N, dropout_dot_product = dropout_dot_product,
                          dropout_fc = dropout_fc, dropout_pos_ff = dropout_pos_ff,
                          old_transformer_model = old_transformer_model) for i in range(n_layers)])

    def get_attention_maps_data(self):

        attentions = {}

        for i, layer in enumerate(self.encoder_layers):
            attentions["layer_{}".format(i + 1)] = layer.multi_head_attention.attention

        return attentions

    def return_attention_maps(self, state):
        self.collect_attention_maps = state
        if state:
            for layer in self.encoder_layers:
                layer.return_attention_maps(state)

    def set_iteration(self, i_iteration):
        if self.collect_attention_maps:
            for layer in self.encoder_layers:
                layer.set_iteration(i_iteration)

    def forward(self, sequence):
        self.batch_size = sequence.shape[0]

        assert (self.batch_size, self.N, self.d_model) == sequence.shape
        encoder_output = self.encoder_layers(sequence)

        return encoder_output
