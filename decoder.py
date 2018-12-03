import torch
import torch.nn as nn
from bahdanau_attention import BahdanauAttention
import torch.nn.functional as F
import numpy
class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True, inputfeeding=False, soft=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.inputfeeding = inputfeeding
        self.soft = soft
        #
        self.rnn_feed = nn.LSTM(emb_size + 2 * hidden_size, hidden_size, num_layers,
                           batch_first=True,  dropout=dropout)
        self.rnn_nofeed = nn.LSTM(emb_size, hidden_size, num_layers,
                    batch_first=True,  dropout=dropout)
        # self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
        #                    batch_first=True,  dropout=dropout)

        self.bridgeh = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        self.bridgec = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)
        self.pre_output_layer_hard = nn.Linear(hidden_size + emb_size,
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

    def decode_soft(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden=None, cell=None, max_len=None):
        if hidden is None:
            hidden = self.init_hidden(encoder_final[0])
        
        if cell is None:
            cell = self.init_cell(encoder_final[1])
        
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
        return decoder_states, (hidden, cell), pre_output_vectors

    def forward(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden=None, cell=None, max_len=None):
        
        if max_len is None:
            max_len = trg_mask.size(-1)
        if self.soft:
            return self.decode_soft(trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden, cell, max_len)
        return self.decode_hard(trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden, cell, max_len)

    def init_hidden(self, encoderh_final):
        if encoderh_final is None:
            return None
        
        return torch.tanh(self.bridgeh(encoderh_final))

    def init_cell(self, encoderc_final):
        if encoderc_final is None:
            return None
        
        return torch.tanh(self.bridgec(encoderc_final))


    def decode_hard(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden=None, cell=None, max_len=None):
        if hidden is None:
            hidden = self.init_hidden(encoder_final[0])
            # 
        if cell is None:
            cell = self.init_cell(encoder_final[1])
        # proj_key = self.attention.key_layer(encoder_hidden)
        decoder_states = []
        pre_output_vectors = []
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, cell, pre_output = self.forward_step_hard(
                prev_embed, encoder_hidden, src_mask, hidden, cell, decoder_states)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, (hidden, cell), pre_output_vectors

    def forward_step_hard(self, prev_embed, encoder_hidden, src_mask, hidden, cell, decoder_states):
        output, (hidden, cell) = self.rnn_nofeed(prev_embed, (hidden, cell))
        decoder_states.append(output)
        query = hidden[-1].unsqueeze(1)
        atten_probs, encoder_proj = self.attention(query=query,#torch.cat(decoder_states, dim=1),
            value=encoder_hidden, mask=src_mask)
        # decoder_hidden = torch.FloatTensor(decoder_states)
        # print decoder_hidden.shape
        outputs = None
        for i in range(len(encoder_proj[0])):
            # if i < len(decoder_states):
                #output
            temp_out = torch.tanh(decoder_states[-1].squeeze(1) + encoder_proj[:,i])
            # else:
            #     temp_out = torch.tanh(decoder_states[-1].squeeze(1).fill_(0) + encoder_proj[:,i])
            
            #logsoftmax
            temp_out = F.softmax(temp_out, dim=-1)
            # pre_output = pre_output.unsqueeze(0)
            temp_out = temp_out.unsqueeze(1)
            if outputs is None:
                outputs = temp_out
            else:
                outputs = torch.cat((outputs, temp_out), dim=1)

        # outputs = F.softmax(outputs, dim=-1)
        final_out = torch.bmm(atten_probs, outputs)
        final_out = torch.log(final_out)
        #log(final_out)
        # for i in range(len(pre_outputs)):
        #     print "a"
        #     print atten_probs.shape
        #     print pre_outputs.shape
            # temp_outputs = torch.bmm(atten_probs, pre_outputs[i])
            # if final_out is None:
            #     final_out = pre_outputs
            # else:
            #     final_out = torch.cat((final_out,temp_outputs), dim=0)
        final_out = torch.cat([prev_embed, final_out], dim=2)
        final_out = self.dropout_layer(final_out)
        final_out = self.pre_output_layer_hard(final_out)
        return output, hidden, cell, final_out
        