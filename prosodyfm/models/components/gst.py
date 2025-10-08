import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class GST(nn.Module):

    def __init__(self, 
                 gst_params):
        super().__init__()
        self.ref_encoder = ReferenceEncoder(gst_params)
        self.stl = STL(gst_params)

    def forward(self, inputs, lengths):
        #print('input x shaape of gst', inputs.shape)  [sum_last_word_num, seq_len]
        #print('input lengths shaape of gst', lengths.shape)  [sum_last_word_num]
        enc_out = self.ref_encoder(inputs, lengths)
        style_embed = self.stl(enc_out)

        return style_embed


class ReferenceEncoder(nn.Module):
    def __init__(self, 
                 gst_params
                ):
        super().__init__()
        self.input_size = 1
        self.hidden_size = gst_params.hidden_size # 128
        self.num_layers = gst_params.num_layers # 1
        self.output_size = gst_params.out_size 

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)  

    def forward(self, x, lengths):
        # x -- pitch contour, [sum_last_word_num, seq_len, 1]   seq_len is the length of the longest sequence in the batch (after padding)
        # lengths -- real lengths of each sequence in the batch, [sum_last_word_num]
        x = x.unsqueeze(-1)
        lengths = lengths.cpu()
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        last_hidden = hidden[-1]

        ref_emb = self.fc(last_hidden)
        return ref_emb

class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    num_units 384
    token_num 6
    num_heads 8
    '''

    def __init__(self, 
                 gst_params):

        super().__init__()
        self.num_units = gst_params.num_units  # num_units // 2 = n_feats
        self.num_heads = gst_params.num_heads  # 8
        self.token_num = gst_params.token_num  

        self.embed = nn.Parameter(torch.FloatTensor(self.token_num, self.num_units // self.num_heads))
        d_q = self.num_units // 2
        d_k = self.num_units // self.num_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=self.num_units, num_heads=self.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  
        #print('query:', query.size()) [3, 1, 192]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  
        #print('keys:', keys.size())  [3, 6, 48]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units//2, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units//2, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units//2, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [3, 1, 192]
        keys = self.W_key(key)   # [3, 6, 192]
        values = self.W_value(key)  # [3, 6, 192]

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [4, 3, 1, 48]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [4, 3, 6, 48]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [4, 3, 6, 48]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [4, 3, 1, 6]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        #print('scores:', scores)
        #print('scores shape', scores.shape)
        #print('values', values)
        #print('values shape', values.shape)
        # out = score * V
        out = torch.matmul(scores, values)  # [4, 3, 1, 48]
        #print('out before:', out)
        #print('out shape before:', out.shape)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [3, 1, 192]
        #print('style_embed:', out)
        #print('style_embed shape:', out.shape)

        return out