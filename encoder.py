import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

    # The input minibatch x should be sorted by length and 
    # have dimensions [batch, time, dim]
    def forward(self, x, mark, lengths):

        packed = pack_padded_sequence(x, lengths, batch_first=True)
        # TODO: difference between h_n and c_n
        # LSTM
        output, (h_final, c_final) = self.rnn(packed)
        # GRU
        # output, h_final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # TODO: how is this bidirectional
        fwdh_final = h_final[0:h_final.size(0):2] #
        bwdh_final = h_final[1:h_final.size(0):2]
        h_final = torch.cat([fwdh_final, bwdh_final], dim=2)
        fwdc_final = c_final[0:c_final.size(0):2] #
        bwdc_final = c_final[1:c_final.size(0):2]
        c_final = torch.cat([fwdc_final, bwdc_final], dim=2)
        return output, h_final, c_final