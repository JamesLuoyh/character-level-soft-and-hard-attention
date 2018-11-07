import torch
import torch.nn as nn
import init
class Batch:
    # TODO: why we need mask, what is pad_index
    def __init__(self, src, trg, pad_index=0):
        src, src_lengths = src
        # print "src pre"
        # print src
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        # print self.src_mask
        
        # print "trgprev"
        # print trg
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        # TODO: why init token for src is None
        if trg is not None:
            trg, trg_lengths = trg
            # TODO: why leave out the last token
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
            # print self.trg
            # print "trgmask"
            # print self.trg_mask
        if init.USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()
            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()
        # exit()