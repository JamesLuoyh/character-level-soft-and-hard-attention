# %matplotlib inline
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython.core.debugger import set_trace

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None, decoder_cell=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden, cell=decoder_cell)
    
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# class Encoder(nn.Module):
#     """Encodes a sequence of word embeddings"""
#     def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
#         super(Encoder, self).__init__()
#         self.num_layers = num_layers
#         self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
#                           batch_first=True, bidirectional=True, dropout=dropout)
        
#     def forward(self, x, mask, lengths):
#         """
#         Applies a bidirectional GRU to sequence of embeddings x.
#         The input mini-batch x needs to be sorted by length.
#         x should have dimensions [batch, time, dim].
#         """
#         packed = pack_padded_sequence(x, lengths, batch_first=True)
#         output, final = self.rnn(packed)
#         output, _ = pad_packed_sequence(output, batch_first=True)

#         # we need to manually concatenate the final states for both directions
#         fwd_final = final[0:final.size(0):2]
#         bwd_final = final[1:final.size(0):2]
#         final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

#         return output, final
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
        return output, (h_final, c_final)


class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        # self.inputfeeding = inputfeeding
        #
        self.rnn_feed = nn.LSTM(emb_size + 2 * hidden_size, hidden_size, num_layers,
                           batch_first=True,  dropout=dropout)
        self.rnn_nofeed = nn.LSTM(emb_size, hidden_size, num_layers,
                    batch_first=True,  dropout=dropout)
        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
                           batch_first=True,  dropout=dropout)

        self.bridgeh = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        self.bridgec = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, hidden, cell,
                     context):
        

        # input feeding
        # context = torch.zeros(hidden.shape[0], encoder_hidden)
        # if self.inputfeeding:
        #     rnn_input = torch.cat([prev_embed, context], dim=2)
        #     output, (hidden, cell) = self.rnn_feed(rnn_input, (hidden, cell))
        # else:
        query = hidden[-1].unsqueeze(1)
        context, atten_probs = self.attention(query=query,
            value=encoder_hidden, mask=src_mask)
        # rnn_input = prev_embed
        # output, (hidden, cell) = self.rnn_nofeed(rnn_input, (hidden, cell))
        
        rnn_input = torch.cat([prev_embed, context], dim=2)
        # output, hidden= self.rnn(rnn_input, hidden)
        output, (hidden, cell) = self.rnn_feed(rnn_input, (hidden, cell))
        pre_output = torch.cat([prev_embed, output, context], dim=2) 
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, cell, context, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden=None, cell=None, max_len=None):
        
        if max_len is None:
            max_len = trg_mask.size(-1)

        if hidden is None:
            hidden = self.init_hidden(encoder_final[0])
        
        if cell is None:
            cell = self.init_cell(encoder_final[1])
        

        # proj_key = self.attention.energy_layer(encoder_hidden)
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

    def init_hidden(self, encoderh_final):
        if encoderh_final is None:
            return None
        
        return torch.tanh(self.bridgeh(encoderh_final))

    def init_cell(self, encoderc_final):
        if encoderc_final is None:
            return None
        
        return torch.tanh(self.bridgec(encoderc_final))

# class Decoder(nn.Module):
#     """A conditional RNN decoder with attention."""
    
#     def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
#                  bridge=True):
#         super(Decoder, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.attention = attention
#         self.dropout = dropout
                 
#         self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers,
#                           batch_first=True, dropout=dropout)
                 
#         # to initialize from the final encoder state
#         self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

#         self.dropout_layer = nn.Dropout(p=dropout)
#         self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
#                                           hidden_size, bias=False)
        
#     def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
#         """Perform a single decoder step (1 word)"""

#         # compute context vector using attention mechanism
#         query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
#         context, attn_probs = self.attention(
#             query=query, proj_key=proj_key,
#             value=encoder_hidden, mask=src_mask)

#         # update rnn hidden state
#         rnn_input = torch.cat([prev_embed, context], dim=2)
#         output, hidden = self.rnn(rnn_input, hidden)
        
#         pre_output = torch.cat([prev_embed, output, context], dim=2)
#         pre_output = self.dropout_layer(pre_output)
#         pre_output = self.pre_output_layer(pre_output)

#         return output, hidden, pre_output
    
#     def forward(self, trg_embed, encoder_hidden, encoder_final, 
#                 src_mask, trg_mask, hidden=None, max_len=None):
#         """Unroll the decoder one step at a time."""
                                         
#         # the maximum number of steps to unroll the RNN
#         if max_len is None:
#             max_len = trg_mask.size(-1)

#         # initialize decoder hidden state
#         if hidden is None:
#             hidden = self.init_hidden(encoder_final)
        
#         # pre-compute projected encoder hidden states
#         # (the "keys" for the attention mechanism)
#         # this is only done for efficiency
#         proj_key = self.attention.key_layer(encoder_hidden)
        
#         # here we store all intermediate hidden states and pre-output vectors
#         decoder_states = []
#         pre_output_vectors = []
        
#         # unroll the decoder RNN for max_len steps
#         for i in range(max_len):
#             prev_embed = trg_embed[:, i].unsqueeze(1)
#             output, hidden, pre_output = self.forward_step(
#               prev_embed, encoder_hidden, src_mask, proj_key, hidden)
#             decoder_states.append(output)
#             pre_output_vectors.append(pre_output)

#         decoder_states = torch.cat(decoder_states, dim=1)
#         pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
#         return decoder_states, hidden, pre_output_vectors  # [B, N, D]

#     def init_hidden(self, encoder_final):
#         """Returns the initial decoder state,
#         conditioned on the final encoder state."""

#         if encoder_final is None:
#             return None  # start with zeros

#         return torch.tanh(self.bridge(encoder_final))

class SoftAttention(nn.Module):

    def __init__(self, hidden_size, encoder_size=None, query_size=None):
        super(SoftAttention, self).__init__()

        # assume bidirectional
        encoder_size = 2 * hidden_size if encoder_size is None else encoder_size
        query_size = hidden_size if query_size is None else query_size
        self.encoder_layer = nn.Linear(encoder_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        # self.attention_layer = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)
        self.attention_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query=None, value=None, mask=None):
        assert mask is not None, "mask is required"
        encoder_proj = self.encoder_layer(value)
        #[B, S, H]
        query_proj = self.query_layer(query)
        #B, 1, H]

        # Calculate scores.
        # TODO: is this correct?
        # expand
        # query_proj = torch.cat([query_proj]*encoder_proj.size(1), 1)
        # query_proj = query_proj.expand(-1, encoder_proj.size(1), -1).contiguous()
        # scores = self.attention_layer(query_proj, encoder_proj)
        scores = self.attention_layer(torch.tanh(query_proj + encoder_proj))
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

class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = SoftAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab))

    return model.cuda() if USE_CUDA else model

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()

def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))

def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(
          np.random.randint(1, num_words, size=(batch_size, length)))
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm

def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None
    cell = None
    for i in range(max_len):
        with torch.no_grad():
            out, (hidden, cell), pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden, cell)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)
  

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]


def print_examples(example_iter, model, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()
        
        count += 1
        if count == n:
            break

def train_copy_task():
    """Train the simple copy task."""
    num_words = 11
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=32, hidden_size=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(num_words=num_words, batch_size=1, num_batches=100))
 
    dev_perplexities = []
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(10):
        
        print("Epoch %d" % epoch)

        # train
        model.train()
        data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad(): 
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_examples(eval_data, model, n=2, max_len=9)
        
    return dev_perplexities

dev_perplexities = train_copy_task()

def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    
plot_perplexity(dev_perplexities)
