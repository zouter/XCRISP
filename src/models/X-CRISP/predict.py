import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model import load_model, FEATURE_SETS, predict, NeuralNetwork
from features import get_features
from indels import gen_indels_v3
from Bio.Seq import Seq
import os


def predict_indels(seq, cutsite, sample_name = ""):
    seq = seq[cutsite-30:cutsite+30]

    if len(seq) < 60:
        print("sequence should have 30nt to left and right of cutsite")
        return

    if seq[33:36] not in ["AGG", "CGG", "GGG", "TGG"]:
        print("sequence should have NNG sequence 3 downstream of cutsite")
        return
    
    indels = gen_indels_v3(seq, 30)
    indels = indels[indels.Type == "DELETION"]
    features = get_features(indels)

    X = features.set_index(indels.Indel)[FEATURE_SETS["v2"]]

    model = load_model("v2", "Base")
    y = predict(model, X)

    print_excel_file(y, indels.Indel.to_list(), seq, sample_name)
    plot_predictions(y, indels.Indel, seq, sample_name)

    return indels, features, y, seq


def generate_indel_sequence(indel, seq, cutsite):
    (pos, size) = indel.split("+")
    start = cutsite+int(pos)
    size = int(size)
    indel_seq = seq[:start] + ("-" * size) + seq[start+size:] 
    indel_seq = indel_seq[:cutsite] + "|" + indel_seq[cutsite:]
    return indel_seq

def print_excel_file(y_pred, indels, seq, sample_name):
    data = [0] + list(y_pred)
    genotypes = [seq[:30] + "|" + seq[30:]] + [generate_indel_sequence(i, seq, 30) for i in indels]
    indels = ["Reference"] + indels
    df = pd.DataFrame({
        "Indel": indels,
        "Deletion Genotype": genotypes,
        "Predicted %": data,
    })
    df.to_excel(os.environ["OUTPUT_DIR"] +"model_predictions/X-CRISP/" + "{}_predictions.xlsx".format(sample_name)) 

def plot_predictions(y_pred, indels, seq, sample_name):
    top_n_indels = 30
    i = np.argsort(y_pred)[::-1][:top_n_indels][::-1]

    # select top 10 indels
    y_pred = y_pred[i]
    indels = indels[i]

    fig, ax = plt.subplots(figsize=(15, 12))

    x = np.arange(len(indels) + 1)
    width = 0.35

    data = list(y_pred) + [0]
    labels = ["({}) {}".format(i, generate_indel_sequence(i, seq, 30)) for i in indels] + ["(Reference) " + seq[:30] + "|" + seq[30:]]

    ax.barh(x, data, align='center', alpha=0.5, height=width, color="C1", label='Predicted %')
    ax.set_xlabel('Percentage')
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontname="monospace")
    ax.legend()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    title = "Predictions for {}".format(sample_name)
    fig.suptitle(title, size=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    img_file = "{}_predictions.png".format(sample_name)
    plt.savefig(os.environ["OUTPUT_DIR"] +"model_predictions/X-CRISP/" + img_file, transparent=False, facecolor='w', edgecolor='w')
    print("Saved plot to {}".format(img_file))

p = "CGATTAGTGAACGGATCGGCACTGCGTGCGCCAATTCTGCAGACAAATGGCAGTATTCATCCACAATTTTAAAAGAAAAGGGGGGATTGGGGGGTACAGTGCAGGGGAAAGAATAGTAGACATAATAGCAACAGACATACAAACTAAAGAATTACAAAAACAAATTACAAAAATTCAAAATTTTCGGGTTTATTACAGGGACAGCAGAGATCCAGTTTGGTTAGTACCGGGCCCTACGCGTCCAAGGTCGGGCAGGAAGAGGGCCTATTTCCCATGATTCCTTCATATTTGCATATACGATACAAGGCTGTTAGAGAGATAATTAGAATTAATTTGACTGTAAACACAAAGATATTAGTACAAAATACGTGACGTAGAAAGTAATAATTTCTTGGGTAGTTTGCAGTTTTAAAATTATGTTTTAAAATGGACTATCATATGCTTACCGTAACTTGAAAGTATTTCGATTTCTTGGCTTTA"
p581_cutsite = 257
p582_cutsite = 232
p583_cutsite = 237

p_rev = Seq(p).reverse_complement()
p_rev582_cutsite = len(p) - p582_cutsite
p_rev583_cutsite = len(p) - p583_cutsite

predict_indels(p, p581_cutsite, sample_name="p581")
predict_indels(p_rev, p_rev582_cutsite, sample_name="p582")
predict_indels(p_rev, p_rev583_cutsite, sample_name="p583")

print("Done.")
