import init
import torch
import numpy as np
import torch.nn as nn
from batch import Batch
import matplotlib.pyplot as plt
import train
import full_model
import random
class Dataload:

    def __init__(self, maxlen):
        src = []
        trg = []
        self.SOS_TOKEN = "<s>"
        self.EOS_TOKEN = "</s>"
        self.PAD_TOKEN = "<pad>"
        self.maxlen = maxlen
        with open('./dataset/cmudict.dict.txt', 'r') as f:
            for line in f:
                tokens = line.split()
                tempsrc = []
                if len(tokens[0]) > maxlen - 2 or len(tokens) > maxlen - 1:
                    continue
                tempsrc.append(self.SOS_TOKEN)
                tempsrc.extend(list(tokens[0]))
                tempsrc.append(self.EOS_TOKEN)
                while len(tempsrc) < maxlen:
                    tempsrc.append(self.PAD_TOKEN)
                src.append(tempsrc)
                temptrg = []
                temptrg.append(self.SOS_TOKEN)
                temptrg.extend(tokens[1:])
                temptrg.append(self.EOS_TOKEN)
                while len(temptrg) < maxlen:
                    temptrg.append(self.PAD_TOKEN)
                trg.append(temptrg)
        

        self.src_stoi = {}
        self.src_itos = {}
        self.trg_stoi = {}
        self.trg_itos = {}
        self.src_stoi[self.PAD_TOKEN] = 0
        self.src_stoi[self.SOS_TOKEN] = 1
        self.src_stoi[self.EOS_TOKEN] = 2
        self.src_itos[0] = self.PAD_TOKEN
        self.src_itos[1] = self.SOS_TOKEN
        self.src_itos[2] = self.EOS_TOKEN
        counter = 3
        self.src_final = []
        for each in src:
            temp = []
            for char in each:
                if char not in self.src_stoi:
                    self.src_stoi[char] = counter
                    self.src_itos[counter] = char
                    counter += 1
                temp.append(self.src_stoi[char])
            self.src_final.append(temp)
        
        self.src_num = len(self.src_stoi)

        self.trg_stoi[self.PAD_TOKEN] = 0
        self.trg_stoi[self.SOS_TOKEN] = 1
        self.trg_stoi[self.EOS_TOKEN] = 2
        self.trg_itos[0] = self.PAD_TOKEN
        self.trg_itos[1] = self.SOS_TOKEN
        self.trg_itos[2] = self.EOS_TOKEN
        counter = 3
        self.trg_final = []
        for each in trg:
            temp = []
            for phoneme in each:
                if phoneme not in self.trg_stoi:
                    self.trg_stoi[phoneme] = counter
                    self.trg_itos[counter] = phoneme
                    counter += 1
                temp.append(self.trg_stoi[phoneme])
            self.trg_final.append(temp)
        datasize = len(src)
        validatesize = datasize / 10
        testsize = datasize / 10
        trainsize = datasize - validatesize - testsize


        self.train_src = self.src_final[0:trainsize]
        self.train_trg = self.src_final[0:trainsize]
        
        self.trg_num = len(self.trg_stoi)
        self.validate_src = self.src_final[trainsize:trainsize+validatesize]
        self.validate_trg = self.src_final[trainsize:trainsize+validatesize]
        self.test_src = self.src_final[-testsize:]
        self.test_trg = self.src_final[-testsize:]



    def data_gen(self, batch_size=16, num_batches=100, eval=False):
        batches = []
        datasrc = self.train_src
        datatrg = self.train_trg
        if eval:
            datasrc = self.validate_src
            datatrg = self.validate_trg
        for i in range(num_batches):
            sample = random.sample(xrange(len(datasrc)), batch_size)
            datasrcbatch = torch.LongTensor([ datasrc[j] for j in  sample])
            datatrgbatch = torch.LongTensor([ datatrg[j] for j in  sample])
            datasrcbatch = datasrcbatch.cuda() if init.USE_CUDA else datasrcbatch
            datatrgbatch = datatrgbatch.cuda() if init.USE_CUDA else datatrgbatch
            src_lengths = [self.maxlen] * batch_size
            trg_lengths = [self.maxlen] * batch_size
            yield Batch((datasrcbatch, src_lengths), (datatrgbatch, trg_lengths), pad_index=0)
        