import torch
import torch.nn as nn
import torch.nn.functional as F
class SoftAttention(nn.Module):

    def __init__(self, hidden_size, encoder_size=None, query_size=None):
        super(SoftAttention, self).__init__()

        # assume bidirectional
        encoder_size = 2 * hidden_size if encoder_size is None else encoder_size
        query_size = hidden_size if query_size is None else query_size
        self.encoder_layer = nn.Linear(encoder_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.attention_layer = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)

    def forward(self, query=None, value=None, mask=None):
        assert mask is not None, "mask is required"
        encoder_proj = self.encoder_layer(value)
        #[B, S, H]
        query_proj = self.query_layer(query)
        #B, 1, H]

        # Calculate scores.
        # TODO: is this correct?
        # expand
        query_proj = torch.cat([query_proj]*encoder_proj.size(1), 1)

        scores = self.attention_layer(query_proj, encoder_proj)
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        # print scores
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        return context, alphas