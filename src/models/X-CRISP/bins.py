# function to aggregate the different outcomes to produce a new set of labels y
import torch
import numpy as np

DEFAULT_BINS = [1, 2, 3, 4, 5, list(range(6, 10)), list(range(10, 15)), list(range(15, 20)), 20]

def get_range_label(b):
    return "{}-{}".format(min(b), max(b))

def get_bin_labels(bins=DEFAULT_BINS):
    labels = []
    for b in bins: 
        if type(b) == int: 
            labels.append(str(b))
        if type(b) == list:
            labels.append(get_range_label(b))
    return labels

def get_bin(x, bins):
    for i, b in enumerate(bins): 
        if type(b) == int: 
            if x == b: 
                return i, b
        if type(b) == list:
            if x in b:
                return i, get_range_label(b)
    return (i, b if type(b) == int else get_range_label(b))

def bin_repair_outcomes_by_length(fractions, sizes, bins=DEFAULT_BINS):
    output = torch.zeros(len(bins), requires_grad=True)
    for i, s in enumerate(sizes):
        binidx, binname = get_bin(s, bins)
        mask = torch.cat((torch.zeros(binidx), 
            fractions[i].reshape(-1), 
            torch.zeros(len(bins) - binidx - 1)))
        output = output + mask
    return output, get_bin_labels(bins)

def bin_repair_outcomes_by_length_numpy(fractions, sizes, bins=DEFAULT_BINS):
    output = np.zeros(len(bins))
    for i, s in enumerate(sizes):
        binidx, binname = get_bin(s, bins)
        mask = np.concatenate((np.zeros(binidx), 
            fractions[i].reshape(-1), 
            np.zeros(len(bins) - binidx - 1)))
        output = output + mask
    return output, get_bin_labels(bins)


if __name__ == "__main__":
    fractions = torch.tensor([.1, .2, .2, .1, .3, .05, 0, .025, .025])
    sizes = torch.tensor([1, 1, 2, 3, 2, 10, 10, 10, 20])
    output, bins = bin_repair_outcomes_by_length(fractions, sizes)
    assert(len(output) == len(bins)) 
    print(output, bins)

    fractions = np.array([.1, .2, .2, .1, .3, .05, 0, .025, .025])
    sizes = np.array([1, 1, 2, 3, 2, 10, 10, 10, 20])
    output, bins = bin_repair_outcomes_by_length_numpy(fractions, sizes)
    assert(len(output) == len(bins)) 
    print(output, bins)
