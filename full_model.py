import torch
import torch.nn as nn
import init
from soft_attention import SoftAttention
from hard_attention import HardAttention
from bahdanau_attention import BahdanauAttention
from encoder import Encoder
from decoder import Decoder
from encoder_decoder import EncoderDecoder
from generator import Generator

def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1, inputfeeding=False, soft=True):

    
    attention = SoftAttention(hidden_size)
    if not soft:
        attention = HardAttention(hidden_size)
    # attention = BahdanauAttention(hidden_size)
    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout, inputfeeding=inputfeeding, soft=soft),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab)
    )

    return model.cuda() if init.USE_CUDA else model