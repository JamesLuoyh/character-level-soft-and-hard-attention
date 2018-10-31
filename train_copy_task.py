import init
import torch
import numpy as np
import torch.nn as nn
from batch import Batch
import matplotlib.pyplot as plt
import train
import full_model
def train_copy_task():
    num_words = 11

    criterion = nn.NLLLoss(ignore_index=0, reduction='sum')
    model = full_model.make_model(num_words, num_words, emb_size=32, hidden_size=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(num_words=num_words, batch_size=1, num_batches=100))

    dev_perplexities = []

    if init.USE_CUDA:
        model.cuda()
    for epoch in range(10):
        print("Epoch %d" % epoch)

        model.train()
        data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        train.run_epoch(data, model,
                  train.SimpleLossCompute(model.generator, criterion, optim))
        
        model.eval()
        with torch.no_grad():
            perplexity = train.run_epoch(eval_data, model,
                                   train.SimpleLossCompute(model.generator, criterion, None))
            
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            
    return dev_perplexities


def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Simple data for copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(
            np.random.randint(1, num_words, size=(batch_size, length))
        )

        data[:, 0] = sos_index
        data = data.cuda() if init.USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)



dev_perplexities = train_copy_task()

def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    plt.show()
    
plot_perplexity(dev_perplexities)