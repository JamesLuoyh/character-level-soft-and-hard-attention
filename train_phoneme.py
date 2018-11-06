import init
import torch
import numpy as np
import torch.nn as nn
from batch import Batch
import matplotlib.pyplot as plt
import train
import full_model
from dataload import Dataload
import argparse
import print_example as print_e
# import sacrebleu
def train_phoneme(num_layers, lr, batch_size, hidden,
                  numepoch, dropout, inputfeeding, cuda, maxlen):
    dataset = Dataload(maxlen=maxlen)
    criterion = nn.NLLLoss(ignore_index=0, reduction='sum')
    model = full_model.make_model(dataset.src_num, dataset.trg_num, emb_size=32,
                                  hidden_size=hidden, num_layers=num_layers, dropout=dropout,
                                  inputfeeding=inputfeeding)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    eval_data = list(dataset.data_gen(batch_size=1, num_batches=100, eval=True))
    dev_perplexities = []

    if init.USE_CUDA and cuda:
        model.cuda()
    for epoch in range(numepoch):
        print("Epoch %d" % epoch)

        model.train()
        data = dataset.data_gen(batch_size=batch_size, num_batches=100)
        train.run_epoch(data, model,
                  train.SimpleLossCompute(model.generator, criterion, optim))
        model.eval()
        with torch.no_grad():
            perplexity = train.run_epoch(eval_data, model,
                                   train.SimpleLossCompute(model.generator, criterion, None))
            
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_e.print_examples(eval_data, dataset, model, n=2, max_len=maxlen)
    return dev_perplexities

def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputfeeding", action="store_true",
                        help="Whether we use input feeding")
    parser.add_argument("--attention", type=str,
                        default='soft',
                        help="Input soft or hard for the attention model")
    parser.add_argument("--cuda", action="store_true",
                        help="Whether we want to use GPU")
    parser.add_argument("--dropout", type=float,
                        default=0.1,
                        help="Dropout")
    parser.add_argument("--lr", type=float,
                        default=0.0003,
                        help="Learning rate")
    parser.add_argument("--layer", type=int,
                        default=1,
                        help="Number of layers")
    parser.add_argument("--batchsize", type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument("--epoch", type=int,
                        default=10,
                        help="Number of epoch")
    parser.add_argument("--hidden", type=int,
                        default=64,
                        help="Hidden layer size")
    parser.add_argument("--maxlen", type=int,
                        default=64,
                        help="Maxlen of a sequence")
    args = parser.parse_args()
    dev_perplexities = train_phoneme(args.layer, args.lr, args.batchsize,
                  args.hidden, args.epoch, args.dropout,
                  args.inputfeeding, args.cuda, args.maxlen)
    plot_perplexity(dev_perplexities)