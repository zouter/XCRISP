# %%
import torch
import numpy as np
import pandas as pd
from tensorflow import keras
import polyptich as pp

pp.setup_ipython()

import os

os.environ["OUTPUT_DIR"] = "./"

from src.models.XCRISP.deletion import load_model, NeuralNetwork, FEATURE_SETS
from src.models.XCRISP.indels import gen_indels_v3
from src.models.XCRISP.features import get_features, get_insertion_features
from src.models.Lindel.features import onehotencoder

# %%
INSERTION_MODEL_F = "./models/Lindel/100x_insertion.h5"
INDEL_MODEL_F = "./models/Lindel/100x_indel.h5"
XCRISPR_MODEL_D = "./models/XCRISP/"

guide = {
    "TargetSequence": "CTCCTATAATTCTAATCACTACAAGTCAGGAATGCCTGCGTTTGGCCGTCCAGTTAGTAACAGAAGGTCAGGTAAGAGG",
    "PAM Index": 42,
}

# Acvrl1_1
guide = {
    "TargetSequence": "GCCCCAGTTGGCCCTGAGGCTAGCTGTGTCCGCGGCCTGCGGCCTGGCGCACCTACATG",
    "PAM Index": 39,
}

# Acvrl1_2
guide = {
    "TargetSequence": "TAGGATGTTGTCGTGTCTAAGCAGAACTGTGTTGTAGATCTCCGTCTCCCGGAACCAGGACTGCTCATCTCGTGAGGAG",
    "PAM Index": 49,
}

# Mitf_3
guide = {
    "TargetSequence": "GCGGTGGCAGGCCCTGGTTGCTGTAGAGGTCGATCAAGTTTCCAGAGACGGGTAACTAGACACAGACAAGAACAGCAAT",
    "PAM Index": 49,
}


# %%
print(
    guide["TargetSequence"][: (guide["PAM Index"] - 3)],
    guide["TargetSequence"][(guide["PAM Index"] - 3) :],
)

# %%
# load models
deletion_model = load_model(
    model_dir=XCRISPR_MODEL_D, loss_fn="kld", learning_rate="0.05"
)
insertion_model = keras.models.load_model(INSERTION_MODEL_F)
indel_model = keras.models.load_model(INDEL_MODEL_F)
print("Models loaded")

# get features
cutsite = guide["PAM Index"] - 3
seq = guide["TargetSequence"][cutsite - 32 : cutsite + 32]
indels = gen_indels_v3(seq, 32, max_deletion_length=30).set_index(["Indel"])
del_features = get_features(indels[indels["Type"] == "DELETION"])
ins_features = get_insertion_features(guide["TargetSequence"], guide["PAM Index"] - 3)
print("Created features")

# predict
with torch.no_grad():
    # deletion predictions from our Model
    x = torch.tensor(del_features[FEATURE_SETS["v4"]].to_numpy()).float()
    ds = deletion_model(x)
    ds = ds / sum(ds)
    ds = ds.detach().numpy()[:, 0]

# insertion predictions from Lindel
seq = guide["TargetSequence"][guide["PAM Index"] - 33 : guide["PAM Index"] + 27]
pam = {"AGG": 0, "TGG": 0, "CGG": 0, "GGG": 0}
guide = seq[13:33]
input_indel = onehotencoder(guide)
input_ins = onehotencoder(guide[-6:])
dratio, insratio = indel_model.predict(np.matrix(input_indel))[0, :]
ins = insertion_model.predict(np.matrix(input_ins))[0, :]

# combine predictions from both models
y_hat = np.concatenate((ds * dratio, ins * insratio), axis=None)

# get labels
ins_labels = [
    "1+A",
    "1+T",
    "1+C",
    "1+G",
    "2+AA",
    "2+AT",
    "2+AC",
    "2+AG",
    "2+TA",
    "2+TT",
    "2+TC",
    "2+TG",
    "2+CA",
    "2+CT",
    "2+CC",
    "2+CG",
    "2+GA",
    "2+GT",
    "2+GC",
    "2+GG",
    "3+X",
]
indels = list(indels.index[indels.Type == "DELETION"]) + ins_labels

results = pd.DataFrame(
    {
        "indel": indels,
        "predicted_frequency": y_hat,
    }
)

# %%
results.sort_values("predicted_frequency", ascending=False).head(10).style.bar()

# %%
import pandas as pd
import re


def parse_indel_length(indel_str):
    """
    Given an indel string like '-6+8', '1+T', '3+X', '0+3',
    returns the number of bases inserted/deleted.
    """
    try:
        left, right = indel_str.split("+", 1)
    except ValueError:
        raise ValueError(f"Indel string '{indel_str}' is not in expected format")

    # If right is purely numeric => length is that number
    if right.isdigit():
        return int(right)

    # Otherwise, take the length of the string after +
    return len(right)


def frameshift_category(indel_str):
    """
    Returns the frameshift category: 0, 1, or 2 (mod 3 of length).
    """
    length = parse_indel_length(indel_str)
    return length % 3


def summarize_frameshifts(df, indel_col="indel", freq_col="predicted_frequency"):
    """
    Takes a DataFrame with an indel column and a frequency column.
    Returns a DataFrame summarizing total frequency per frameshift category.
    """
    df = df.copy()
    df["frameshift"] = df[indel_col].apply(frameshift_category)
    summary = df.groupby("frameshift")[freq_col].sum().reset_index()
    return summary


# %%
summarize_frameshifts(results)

# %%
