import torch
import torch.nn as nn
from bahdanau_attention import BahdanauAttention 
class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True, inputfeeding=False):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.inputfeeding = inputfeeding
        #
        self.rnn_feed = nn.LSTM(emb_size + 2 * hidden_size, hidden_size, num_layers,
                           batch_first=True,  dropout=dropout)
        self.rnn_nofeed = nn.LSTM(emb_size, hidden_size, num_layers,
                    batch_first=True,  dropout=dropout)
        # self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
        #                    batch_first=True,  dropout=dropout)

        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, hidden, cell,
                     context):
        

        # input feeding
        # context = torch.zeros(hidden.shape[0], encoder_hidden)
        if self.inputfeeding:
            rnn_input = torch.cat([prev_embed, context], dim=2)
            output, (hidden, cell) = self.rnn_feed(rnn_input, (hidden, cell))
        else:
            rnn_input = prev_embed
            output, (hidden, cell) = self.rnn_nofeed(rnn_input, (hidden, cell))
        query = hidden[-1].unsqueeze(1)
        context, atten_probs = self.attention(query=query,
            value=encoder_hidden, mask=src_mask)
        pre_output = torch.cat([prev_embed, output, context], dim=2) 
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, cell, context, pre_output

    def forward(self, trg_embed, encoder_hidden, encoderh_final, encoderc_final,
                src_mask, trg_mask, hidden=None, cell=None, max_len=None):
        
        if max_len is None:
            max_len = trg_mask.size(-1)

        if hidden is None:
            hidden = self.init_hidden(encoderh_final)
        
        if cell is None:
            cell = self.init_cell(encoderc_final)
        
        # proj_key = self.attention.key_layer(encoder_hidden)
        decoder_states = []
        pre_output_vectors = []
        context = torch.zeros(encoder_hidden.shape[0], 1, 2 * hidden.shape[-1])
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, cell, context, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, hidden, cell, context
            )
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors

    def init_hidden(self, encoderh_final):
        if encoderh_final is None:
            return None
        
        return torch.tanh(self.bridge(encoderh_final))

    def init_cell(self, encoderc_final):
        if encoderc_final is None:
            return None
        
        return torch.tanh(self.bridge(encoderc_final))
