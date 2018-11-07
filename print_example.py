import torch
import numpy as np
from dataload import Dataload
def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_finalh, encoder_finalc = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_finalh, encoder_finalc, src_mask,
              prev_y, trg_mask, hidden)

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
        x = [vocab[i] for i in x]

    return [str(t) for t in x]

def print_examples(example_iter, dataload, model,
                   n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=2, 
                   trg_eos_index=2):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print
    
    if dataload.src_stoi is not None and dataload.trg_stoi is not None:
        src_eos_index = dataload.src_stoi[dataload.EOS_TOKEN]
        trg_sos_index = dataload.trg_stoi[dataload.SOS_TOKEN]
        trg_eos_index = dataload.trg_stoi[dataload.EOS_TOKEN]
    else:
        src_eos_index = 2
        trg_sos_index = 1
        trg_eos_index = 2
        
    for i, batch in enumerate(example_iter):
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]
        # print src
        # print trg
        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=dataload.src_itos)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=dataload.trg_itos)))
        print("Pred: ", " ".join(lookup_words(result, vocab=dataload.trg_itos)))
        print
        
        count += 1
        if count == n:
            break