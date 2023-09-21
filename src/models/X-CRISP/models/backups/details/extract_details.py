import torch

t = torch.load("Sigmoid_1000x_full_BaseLoss_RS_42_model.details")

samples = t["samples"]

f = open("samples_1000x.txt", "w")
for s in samples:
    f.write(s + "\n")

f.close()