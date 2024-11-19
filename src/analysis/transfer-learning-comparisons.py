from functools import reduce
import random
import sys, os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.special import rel_entr
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from math import log2
from sklearn.metrics import mean_squared_error
from scipy.stats import weightedtau


MIN_NUM_READS = 100
REP2_COUNTS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/replicate_{}.pkl"
mESC_WT_COUNTS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/mESC_WT_{}.pkl"
CRISPRedict_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/model_v4_kld_{}.pkl"
CRISPRedict_MSE_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/model_1_v4_RS_1_{}.pkl"
LINDEL_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/Lindel/predictions_{}x_{}.pkl"
FORECasT_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/FORECasT/predictions_{}x_{}.pkl"
TRANSFER_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/transfer_kld_{}_{}_RS_1_{}.pkl"
TEST_FILES = ["052218-U2OS-+-LibA-postCas9-rep1_transfertest", "0226-PRLmESC-Lib1-Cas9_transfertest", "HAP1_test", "TREX_A_test"]

data = {}
# t = sys.argv[1]

# # if t not in TEST_FILES:
# #     print("select one of", TEST_FILES)
# #     exit()
for i, t in enumerate(TEST_FILES):
    data[t] = {}
    data[t]["CRISPRedict"] = pkl.load(open(CRISPRedict_PREDICTIONS_F.format(t), 'rb'))
    # data[t]["CRISPRedict MSE"] = pkl.load(open(CRISPRedict_MSE_PREDICTIONS_F.format(t), 'rb'))
    # # if os.path.exists(REP2_COUNTS_F.format(t)):
    #     # data[t]["Replicate"] = pkl.load(open(REP2_COUNTS_F.format(t), 'rb'))
    # # if os.path.exists(mESC_WT_COUNTS_F.format(t)):
    # #     data[t]["mESC WT"] = pkl.load(open(mESC_WT_COUNTS_F.format(t), 'rb'))
    data[t]["Lindel"] = pkl.load(open(LINDEL_PREDICTIONS_F.format(MIN_NUM_READS, t), 'rb'))
    data[t]["FORECasT"] = pkl.load(open(FORECasT_PREDICTIONS_F.format(MIN_NUM_READS, t), 'rb'))

    # need to remove psuedocounts that were added to profiles in preparation for training
    for target_site in data[t]["FORECasT"].keys():
        data[t]["FORECasT"][target_site]["actual"] = np.array(data[t]["FORECasT"][target_site]["actual"]) - 0.5

    # print("loaded baselines for " + t)
    # for mode in ["pretrained", "baseline", "pretrainedsamearch", "pretrainedplusonelayer", "pretrainedonefrozenlayer",  "weightinit"]:
    #     for num_samples in [2, 5, 10, 20, 50, 100, 200, 500]:
    #         data[t]["transfer_{}_{}".format(mode, num_samples)] = pkl.load(open(TRANSFER_PREDICTIONS_F.format(mode, num_samples, t), 'rb'))
    # print("loaded transfer models for " + t)

models = list(data[t].keys())

for t in TEST_FILES:
    for e in data[t]:
        print(t, e, len(data[t][e].keys()))

# collect targets common to all experiments

common_oligos = {}
for t in TEST_FILES:
    t1 = np.array(list(data[t]["CRISPRedict"].keys()))
    t3 = np.array(list(data[t]["Lindel"].keys()))
    t4 = np.array(list(data[t]["FORECasT"].keys()))
    common_oligos[t] = reduce(np.intersect1d, (t1, t3, t4))
    print(len(common_oligos[t]))

# reformat FORECasT indels
def FC_indel_to_our(fc):
    parts = fc.split("_")
    t = parts[0][0] # type I or D 
    l = int(parts[0][1:]) # size
    p = int(parts[1].split("R")[1]) # start
    if t == "D":
        return "{}+{}".format(p-l, l)
    else:
        # return "I{}".format(l)
        return fc

for t in TEST_FILES:
    for o in common_oligos[t]:
        data[t]["FORECasT"][o]["indels"] = [FC_indel_to_our(i) for i in data[t]["FORECasT"][o]["indels"]]

file_mapping = {
    "0226-PRLmESC-Lib1-Cas9_transfertest": "inDelphi NHEJ-deficient",
    "052218-U2OS-+-LibA-postCas9-rep1_transfertest": "inDelphi USO2 WT",
    "HAP1_test": "FORECasT HAP1",
    "TREX_A_test": "FORECasT TREX",
    "2A_TREX_A_test": "2A_FORECasT TREX",
}

# test over a common set of indels per target site
def generate_1_and_2_bp_insertions():
    nucs = ["A", "C", "G", "T"]
    onebps = []
    twobps = []
    for n1 in nucs:
        onebps.append("1+" + n1)
        for n2 in nucs:
            twobps.append("2+" + n1 + n2)
    return onebps + twobps

common_insertions = generate_1_and_2_bp_insertions()

print("reformatted FORECasT data")

def generate_1_and_2_bp_insertions():
    nucs = ["A", "C", "G", "T"]
    onebps = []
    twobps = []
    for n1 in nucs:
        onebps.append("1+" + n1)
        for n2 in nucs:
            twobps.append("2+" + n1 + n2)
    return onebps + twobps

common_insertions = generate_1_and_2_bp_insertions()
print("generated common insertions")

# collect predicted data into dataframe
rows = []
indices = []
for t in TEST_FILES:
    test_f = file_mapping[t]
    for method in data[t].keys():
        for target_site in common_oligos[t]:
            if target_site == "Oligo_10007" and method == 'FORECasT':
                print(target_site, method)

            indels = np.array(data[t][method][target_site]["indels"]) 
            predicted = np.array(data[t][method][target_site]["predicted"]).astype(np.float) # Q
            predicted = predicted/sum(predicted)
            observed = np.array(data[t][method][target_site]["actual"]).astype(np.float)
            observed = observed/sum(observed) # P
            indices.append((test_f, method, target_site))
            correlation = np.corrcoef(predicted, observed)[0,1]
            kl_divergence = entropy(observed, predicted)
            js = jensenshannon(observed, predicted)
            rows.append([correlation, kl_divergence, js])

indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
df = df[~df.index.get_level_values(2).isin(inf_oligos)]
df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/overall.tsv", sep="\t")

print("Calculated overall results")

def is_insertion(indel, method):
    if method in ["CRISPRedict", "Lindel", "inDelphi"] or "transfer" in method:
        return indel in common_insertions or indel == "3" or indel == "3+X"
    if method == "FORECasT":
        return indel[0] == "I"

# collect predicted data into dataframe
rows = []
indices = []
for t in TEST_FILES:
    test_f = file_mapping[t]
    for method in data[t].keys():
        for target_site in common_oligos[t]:
            indels = np.array(data[t][method][target_site]["indels"])
            # deletions = np.array([not is_insertion(x, method) for x in indels])
            if method == "CRISPRedict" or "transfer" in method:
                mh = np.array(data[t][method][target_site]["mh"] + ([False] * 21)) 
            else:
                mh = np.array(data[t][method][target_site]["mh"])
            predicted = np.array(data[t][method][target_site]["predicted"])[mh].astype(np.float) # Q
            observed = np.array(data[t][method][target_site]["actual"])[mh].astype(np.float) # P

            predicted = predicted/sum(predicted)
            observed = observed/sum(observed)

            # then calculate
            indices.append((test_f, method, target_site))
            correlation = np.corrcoef(predicted, observed)[0,1]
            kl_divergence = entropy(observed, predicted)
            js = jensenshannon(observed, predicted)
            rows.append([correlation, kl_divergence, js])

indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
df = df[~df.index.get_level_values(2).isin(inf_oligos)]
df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/mh.tsv", sep="\t")

print("Calculated MH results")

# collect predicted data into dataframe
rows = []
indices = []
for t in TEST_FILES:
    test_f = file_mapping[t]
    for method in data[t].keys():
        if method == "inDelphi": continue
        for target_site in common_oligos[t]:
            indels = np.array(data[t][method][target_site]["indels"])
            deletions = np.array([not is_insertion(x, method) for x in indels])
            if method == "CRISPRedict" or "transfer" in method:
                mh = np.array(data[t][method][target_site]["mh"] + ([False] * 21)) 
            else:
                mh = np.array(data[t][method][target_site]["mh"])
            mhless = np.invert(mh)
            mhless_deletions = deletions & mhless
            predicted = np.array(data[t][method][target_site]["predicted"])[mhless_deletions].astype(np.float) # Q
            observed = np.array(data[t][method][target_site]["actual"])[mhless_deletions].astype(np.float) # P

            predicted = predicted/sum(predicted)
            observed = observed/sum(observed)

            # then calculate
            indices.append((test_f, method, target_site))
            correlation = np.corrcoef(predicted, observed)[0,1]
            kl_divergence = entropy(observed, predicted)
            js = jensenshannon(observed, predicted)
            rows.append([correlation, kl_divergence, js])

indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
df = df[~df.index.get_level_values(2).isin(inf_oligos)]
df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/mhless.tsv", sep="\t")

print("Calculated MH Less results")


# collect predicted data into dataframe
rows = []
indices = []
for t in TEST_FILES:
    test_f = file_mapping[t]
    for method in data[t].keys():
        for target_site in common_oligos[t]:
            indels = np.array(data[t][method][target_site]["indels"])
            insertions = np.array([is_insertion(x, method) for x in indels])
            predicted = np.array(data[t][method][target_site]["predicted"])[insertions].astype(np.float) # Q
            observed = np.array(data[t][method][target_site]["actual"])[insertions].astype(np.float) # P

            predicted = predicted/sum(predicted)
            observed = observed/sum(observed)

            # then calculate
            indices.append((test_f, method, target_site))
            correlation = np.corrcoef(predicted, observed)[0,1]
            kl_divergence = entropy(observed, predicted)
            js = jensenshannon(observed, predicted)
            rows.append([correlation, kl_divergence, js])

indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
df = df[~df.index.get_level_values(2).isin(inf_oligos)]
df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/insertions.tsv", sep="\t")

print("Calculated insertion results")

# collect predicted data into dataframe
rows = []
indices = []
for t in TEST_FILES:
    test_f = file_mapping[t]
    for method in data[t].keys():
        p = []
        o = []
        for target_site in common_oligos[t]:
            indels = np.array(data[t][method][target_site]["indels"])
            insertions = np.array([is_insertion(x, method) for x in indels])
            predicted = np.array(data[t][method][target_site]["predicted"]).astype(np.float) # Q
            observed = np.array(data[t][method][target_site]["actual"]).astype(np.float) # P

            p.append(sum(predicted[insertions])/sum(predicted))
            o.append(sum(observed[insertions])/sum(observed))

        # then calculate
        indices.append((test_f, method))
        rows.append([mean_squared_error(p, o)])

indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method"])
df = pd.DataFrame(rows, index=indices, columns=["Mean Squared Error"])
df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/indels.tsv", sep="\t")

print("Calculated indel results")


