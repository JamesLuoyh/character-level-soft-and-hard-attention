import time
import math
import torch

def run_epoch(data_iter, model, loss_compute, print_every=50):

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths
        )

        loss, correct = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f accuracy %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, correct / float(batch.ntokens), print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens)), correct / float(total_tokens)

    
class SimpleLossCompute:

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        correct = 0.0
        _, xpred = torch.max(x, dim=2)
        for i in range(len(xpred)):
            xtrg = xpred[i]
            match = True
            for j in range(len(xtrg)):
                if len(y[i]) <= j or abs(xtrg[j] - y[i][j]) > 0.0000001:
                    match = False
                    break
            if match:
                correct += 1.0

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.data.item() * norm, correct