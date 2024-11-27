from functools import reduce
import random
import sys, os, re
import copy
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.special import rel_entr
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from math import log2
from sklearn.metrics import mean_squared_error
from scipy.stats import weightedtau
from src.data.data_loader import get_details_from_fasta



MIN_NUM_READS = 100
REP2_COUNTS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/replicate_{}.pkl"
mESC_WT_COUNTS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/mESC_WT_{}.pkl"
CRISPRedict_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/model_v4_kld_{}.pkl"
CRISPRedict_MSE_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/model_1_v4_RS_1_{}.pkl"
INDELPHI_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/predictions_{}x_{}_indelphi.pkl"
LINDEL_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/Lindel/predictions_{}x_{}.pkl"
FORECasT_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/FORECasT/predictions_{}x_{}.pkl"
TRANSFER_PREDICTIONS_F = os.environ["OUTPUT_DIR"] + "/model_predictions/OurModel/transfer_kld_{}_{}_RS_1_{}.pkl"
TEST_FILES = ["052218-U2OS-+-LibA-postCas9-rep1_transfertest", "0226-PRLmESC-Lib1-Cas9_transfertest", "HAP1_test", "TREX_A_test"]
# TEST_FILES = ["TREX_A_test"]

data = {}
# t = sys.argv[1]

# # if t not in TEST_FILES:
# #     print("select one of", TEST_FILES)
# #     exit()
# for i, t in enumerate(TEST_FILES):
#     data[t] = {}
#     data[t]["CRISPRedict"] = pkl.load(open(CRISPRedict_PREDICTIONS_F.format(t), 'rb'))
#     # data[t]["CRISPRedict MSE"] = pkl.load(open(CRISPRedict_MSE_PREDICTIONS_F.format(t), 'rb'))
#     # # if os.path.exists(REP2_COUNTS_F.format(t)):
#     #     # data[t]["Replicate"] = pkl.load(open(REP2_COUNTS_F.format(t), 'rb'))
#     # # if os.path.exists(mESC_WT_COUNTS_F.format(t)):
#     # #     data[t]["mESC WT"] = pkl.load(open(mESC_WT_COUNTS_F.format(t), 'rb'))
#     data[t]["Lindel"] = pkl.load(open(LINDEL_PREDICTIONS_F.format(MIN_NUM_READS, t), 'rb'))
#     data[t]["FORECasT"] = pkl.load(open(FORECasT_PREDICTIONS_F.format(MIN_NUM_READS, t), 'rb'))
#     data[t]["inDelphi"] = pkl.load(open(INDELPHI_PREDICTIONS_F.format(MIN_NUM_READS, t), 'rb'))

#     # need to remove psuedocounts that were added to profiles in preparation for training
#     for target_site in data[t]["FORECasT"].keys():
#         data[t]["FORECasT"][target_site]["actual"] = np.array(data[t]["FORECasT"][target_site]["actual"]) - 0.5

#     print("loaded baselines for " + t)
#     for mode in ["pretrained", "baseline", "pretrainedsamearch", "pretrainedplusonelayer", "pretrainedonefrozenlayer",  "weightinit"]:
#         for num_samples in [2, 5, 10, 20, 50, 100, 200, 500]:
#             data[t]["transfer_{}_{}".format(mode, num_samples)] = pkl.load(open(TRANSFER_PREDICTIONS_F.format(mode, num_samples, t), 'rb'))
#     print("loaded transfer models for " + t)

# models = list(data[t].keys())

# for t in TEST_FILES:
#     for e in data[t]:
#         print(t, e, len(data[t][e].keys()))

# # collect targets common to all experiments

# common_oligos = {}
# for t in TEST_FILES:
#     all_t = []
#     all_t.append(np.array(list(data[t]["CRISPRedict"].keys())))
#     all_t.append(np.array(list(data[t]["inDelphi"].keys())))
#     all_t.append(np.array(list(data[t]["Lindel"].keys())))
#     all_t.append(np.array(list(data[t]["FORECasT"].keys())))
#     common_oligos[t] = reduce(np.intersect1d, all_t)
#     print(len(common_oligos[t]))

# def generate_1_and_2_bp_insertions():
#     nucs = ["A", "C", "G", "T"]
#     onebps = []
#     twobps = []
#     for n1 in nucs:
#         onebps.append("1+" + n1)
#         for n2 in nucs:
#             twobps.append("2+" + n1 + n2)
#     return onebps + twobps

# common_insertions = generate_1_and_2_bp_insertions()
# print("generated common insertions")

# # reformat FORECasT indels
# def FC_indel_to_our(fc):
#     parts = fc.split("_")
#     t = parts[0][0] # type I or D 
#     l = int(parts[0][1:]) # size
#     p = int(parts[1].split("R")[1]) # start
#     if t == "D":
#         return "{}+{}".format(p-l, l)
#     else:
#         # return "I{}".format(l)
#         return fc

# def inDelphi_to_our(ind):
#     if ind[-1] in "ACGT":
#         return "1+" + ind[-1]
#     if ind.isnumeric():
#         return "DL" + ind 
#     return ind

# for t in TEST_FILES:
#     for o in common_oligos[t]:
#         data[t]["FORECasT"][o]["indels"] = [FC_indel_to_our(i) for i in data[t]["FORECasT"][o]["indels"]]
#         data[t]["inDelphi"][o]["indels"] = [inDelphi_to_our(i) for i in data[t]["inDelphi"][o]["indels"]]

# print("reformatted FORECasT + inDelphi data")

file_mapping = {
    "0226-PRLmESC-Lib1-Cas9_transfertest": "inDelphi NHEJ-deficient",
    "052218-U2OS-+-LibA-postCas9-rep1_transfertest": "inDelphi USO2 WT",
    "HAP1_test": "FORECasT HAP1",
    "TREX_A_test": "FORECasT TREX",
    "2A_TREX_A_test": "2A_FORECasT TREX",
}

# # FORECasT Mappings
# fasta_files = ["./src/data/FORECasT/test.fasta", "./src/data/inDelphi/LibA.fasta"]
# guides = {}

# for ff in fasta_files:
#     guides.update(get_details_from_fasta(ff))

# ins_mapping = {}
# t = "test"
# for t in TEST_FILES:
#     ins_mapping[t] = {}
#     for o in common_oligos[t]:
#         g = guides[o]
#         cutsite = g["PAM Index"] - 3
#         ins_mapping[t][o] = {}
#         FORECasT_insertions = [i for i in data[t]["FORECasT"][o]["indels"] if "I" in i]
#         FORECasT_rep_insertions = [i for i in FORECasT_insertions if "C" in i]
#         FORECasT_norep_insertions = [i for i in FORECasT_insertions if "C" not in i]
#         if len(FORECasT_norep_insertions) not in [1, 2]: print("wtf")
#         for i in FORECasT_rep_insertions:
#             _, I, _, L, C, R = re.split("I|_|L|C|R", i)
#             rep_nuc = g["TargetSequence"][cutsite + int(R) -int(I):cutsite + int(R)]
#             ins_mapping[t][o][i] = "{}+{}".format(int(I), rep_nuc)
#         for i in FORECasT_norep_insertions:
#             if i[1] == "1":
#                 ins_mapping[t][o][i] = list(np.setdiff1d([c for c in common_insertions if "1" in c], list(ins_mapping[t][o].values()))) 
#             if i[1] == "2":
#                 ins_mapping[t][o][i] = list(np.setdiff1d([c for c in common_insertions if "2" in c], [c for c in list(ins_mapping[t][o].values()) if "2" in c])) 
# print("mapped output to FORECasT")
# print(ins_mapping[t][o])

# rev_ins_mapping = {}
# for t in ins_mapping:
#     rev_ins_mapping[t] = {}
#     for o in ins_mapping[t]:
#         rev_ins_mapping[t][o] = {}
#         for i in ins_mapping[t][o]:
#             a = ins_mapping[t][o][i]
#             if isinstance(a, list):
#                 for a2 in a:
#                     rev_ins_mapping[t][o][a2] = i
#             else:
#                rev_ins_mapping[t][o][a] = i
# print("Reversed mapping")
# print(rev_ins_mapping[t][o])

# def is_insertion(indel, method):
#     if method in ["CRISPRedict", "Lindel", "inDelphi"] or "transfer" in method:
#         return indel in common_insertions or indel == "3" or indel == "3+X"
#     if method == "FORECasT":
#         return indel[0] == "I"

# collect predicted data into dataframe
# rows = []
# indices = []
# for t in TEST_FILES:
#     test_f = file_mapping[t]
#     for method in tqdm(data[t].keys()):
#         print(method)
#         for target_site in common_oligos[t]:
#             if target_site == "Oligo_10007" and method == 'FORECasT':
#                 print(target_site, method)

#             indels = np.array(data[t][method][target_site]["indels"]) 
#             predicted = np.array(data[t][method][target_site]["predicted"]).astype(float) # Q
#             predicted = predicted/sum(predicted)
#             observed = np.array(data[t][method][target_site]["actual"]).astype(float)
#             observed = observed/sum(observed) # P
#             indices.append((test_f, method, target_site))
#             correlation = np.corrcoef(predicted, observed)[0,1]
#             kl_divergence = entropy(observed, predicted)
#             js = jensenshannon(observed, predicted)
#             rows.append([correlation, kl_divergence, js])

# indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
# df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
# inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
# df = df[~df.index.get_level_values(2).isin(inf_oligos)]
# df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/overall.tsv", sep="\t")

# print("Calculated overall results")

# # collect predicted data into dataframe
# rows = []
# indices = []
# for t in TEST_FILES:
#     test_f = file_mapping[t]
#     for method in tqdm(data[t].keys()):
#         print(method)
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"])
#             # deletions = np.array([not is_insertion(x, method) for x in indels])
#             if method == "CRISPRedict" or "transfer" in method:
#                 mh = np.array(data[t][method][target_site]["mh"] + ([False] * 21)) 
#             else:
#                 mh = np.array(data[t][method][target_site]["mh"])
#             predicted = np.array(data[t][method][target_site]["predicted"])[mh].astype(float) # Q
#             observed = np.array(data[t][method][target_site]["actual"])[mh].astype(float) # P

#             predicted = predicted/sum(predicted)
#             observed = observed/sum(observed)

#             # then calculate
#             indices.append((test_f, method, target_site))
#             correlation = np.corrcoef(predicted, observed)[0,1]
#             kl_divergence = entropy(observed, predicted)
#             js = jensenshannon(observed, predicted)
#             rows.append([correlation, kl_divergence, js])

# indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
# df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
# inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
# df = df[~df.index.get_level_values(2).isin(inf_oligos)]
# df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/mh.tsv", sep="\t")

# print("Calculated MH results")

# # collect predicted data into dataframe
# rows = []
# indices = []
# for t in TEST_FILES:
#     test_f = file_mapping[t]
#     for method in tqdm(data[t].keys()):
#         print(method)
#         if method == "inDelphi": continue
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"])
#             deletions = np.array([not is_insertion(x, method) for x in indels])
#             if method == "CRISPRedict" or "transfer" in method:
#                 mh = np.array(data[t][method][target_site]["mh"] + ([False] * 21)) 
#             else:
#                 mh = np.array(data[t][method][target_site]["mh"])
#             mhless = np.invert(mh)
#             mhless_deletions = deletions & mhless
#             predicted = np.array(data[t][method][target_site]["predicted"])[mhless_deletions].astype(float) # Q
#             observed = np.array(data[t][method][target_site]["actual"])[mhless_deletions].astype(float) # P

#             predicted = predicted/sum(predicted)
#             observed = observed/sum(observed)

#             # then calculate
#             indices.append((test_f, method, target_site))
#             correlation = np.corrcoef(predicted, observed)[0,1]
#             kl_divergence = entropy(observed, predicted)
#             js = jensenshannon(observed, predicted)
#             rows.append([correlation, kl_divergence, js])

# indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
# df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
# inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
# df = df[~df.index.get_level_values(2).isin(inf_oligos)]
# df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/mhless.tsv", sep="\t")

# print("Calculated MH Less results")

# def is_forecast_insertion(indel):
#     return indel[:2] == "I1"

# ins_data = copy.deepcopy(data)
# # collect predicted data into dataframe
# rows = []
# indices = []
# for t in TEST_FILES:
#     test_f = file_mapping[t]
#     for method in [m for m in models if m != "FORECasT"]:
#         for target_site in common_oligos[t]:
#             new_predicted = []
#             new_observed = []
#             new_indels = []
#             indels = np.array(data[t][method][target_site]["indels"])
#             predicted = pd.Series(list(data[t][method][target_site]["predicted"]), index=indels)
#             observed = pd.Series(list(data[t][method][target_site]["actual"]), index=indels)

#             predicted = predicted/sum(predicted)
#             observed = observed/sum(observed)

#             for i in ins_mapping[t][target_site]:
#                 if i[:2] == "I1":
#                     new_predicted.append(predicted.loc[ins_mapping[t][target_site][i]].sum())
#                     new_observed.append(observed.loc[ins_mapping[t][target_site][i]].sum())
#                     new_indels.append(i)

#             ins_data[t][method][target_site]["indels"] = np.array(new_indels)
#             ins_data[t][method][target_site]["predicted"] = np.array(new_predicted)
#             ins_data[t][method][target_site]["actual"] = np.array(new_observed)


#     for method in tqdm(data[t].keys()):
#         print(method)
#         for target_site in common_oligos[t]:
#             indels = ins_data[t][method][target_site]["indels"]
#             insertions = np.array([is_forecast_insertion(x) for x in indels])
#             predicted = ins_data[t][method][target_site]["predicted"][insertions].astype(float) # Q
#             observed = ins_data[t][method][target_site]["actual"][insertions].astype(float) # P

#             predicted = predicted/sum(predicted)
#             observed = observed/sum(observed)

#             # then calculate
#             indices.append((test_f, method, target_site))
#             correlation = np.corrcoef(predicted, observed)[0,1]
#             kl_divergence = entropy(observed, predicted)
#             js = jensenshannon(observed, predicted)
#             rows.append([correlation, kl_divergence, js])

# indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method", "Target Site"])
# df = pd.DataFrame(rows, index=indices, columns=["Pearson's Correlation", "KL Divergence", "Jensen Shannon"])
# inf_oligos = df[~np.isfinite(df["KL Divergence"])].index.get_level_values(2)
# df = df[~df.index.get_level_values(2).isin(inf_oligos)]
# df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/insertions.tsv", sep="\t")

# print("Calculated insertion results")

# # collect predicted data into dataframe
# rows = []
# indices = []
# for t in TEST_FILES:
#     test_f = file_mapping[t]
#     for method in tqdm(data[t].keys()):
#         print(method)
#         p = []
#         o = []
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"])
#             insertions = np.array([is_insertion(x, method) for x in indels])
#             predicted = np.array(data[t][method][target_site]["predicted"])[insertions].astype(float) # Q
#             observed = np.array(data[t][method][target_site]["actual"])[insertions].astype(float) # P

#             p.append(sum(predicted[insertions])/sum(predicted))
#             o.append(sum(observed[insertions])/sum(observed))

#         # then calculate
#         indices.append((test_f, method))
#         rows.append([mean_squared_error(p, o)])

# indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method"])
# df = pd.DataFrame(rows, index=indices, columns=["Mean Squared Error"])
# df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/indels.tsv", sep="\t")

# print("Calculated indel results")



from sklearn.metrics import mean_squared_error

# # collect predicted data into dataframe
# def is_1bp_deletion(indel, method):
#     if method in ["CRISPRedict", "Lindel", "FORECasT"] or ("transfer" in method):
#         return indel.split("+")[-1] == "1"
#     if method == "inDelphi":
#         return indel.split("+")[-1] == "1" or indel == "DL1"

# def is_1bp_insertion(indel, method):
#     if (method in ["CRISPRedict", "Lindel", "inDelphi"]) or ("transfer" in method):
#         return indel in ['1+A', '1+C', '1+G', '1+T']
#     if method == 'FORECasT':
#         return indel[:2] == "I1"

# def is_insertion(indel, method):
#     if (method in ["CRISPRedict", "Lindel", "inDelphi"]) or ("transfer" in method):
#         return indel in common_insertions or indel == "3" or indel == "3+X"
#     if method == "FORECasT":
#         return indel[0] == "I"

# def is_frameshift(indel, method, t = "any"):
#     length = None
#     if (method in ["CRISPRedict", "Lindel"]) or ("transfer" in method):
#         if is_insertion(indel, method):
#             length = int(indel[0]) 
#         else:
#             length = int(indel.split("+")[-1]) 
#     if method == "inDelphi":
#         if is_insertion(indel, method):
#             length = int(indel[0])
#         elif indel[:2] == "DL":
#             length = int(indel[2:])
#         else:
#             length = int(indel.split("+")[-1]) 
#     if method == "FORECasT":
#         if is_insertion(indel, method):
#             length = int(indel.split("_")[0][1:])
#         else:
#             length = int(indel.split("+")[-1]) 
        
#     allonebpframeshifts = np.array([1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
#     alltwobpframeshifts = allonebpframeshifts + 1

#     if t == 1:
#         return length in allonebpframeshifts
#     elif t == 2:
#         return length in alltwobpframeshifts
#     else:
#         return length % 3 != 0

# rows = []
# indices = []
# for t in TEST_FILES:
#     test_f = file_mapping[t]
#     for method in tqdm(data[t].keys()):
#         print(method)

#         # deletion frequency
#         # one basepair deletions
#         preddelfreq = []
#         obsdelfreq = []
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"]) 
#             dels = [not is_insertion(x, method) for x in indels]
#             preddelratio = sum(np.array(data[t][method][target_site]["predicted"])[dels])/sum(data[t][method][target_site]["predicted"])
#             preddelfreq.append(preddelratio)
#             obsdelratio = sum(np.array(data[t][method][target_site]["actual"])[dels])/sum(data[t][method][target_site]["actual"])
#             obsdelfreq.append(obsdelratio)
#         delmse = mean_squared_error(preddelfreq, obsdelfreq)


#         # one basepair deletions
#         predonebpdelfreq = []
#         obsonebpdelfreq = []
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"]) 
#             onebpdels = [is_1bp_deletion(x, method) for x in indels]
#             predonebpdelratio = sum(np.array(data[t][method][target_site]["predicted"])[onebpdels])/sum(data[t][method][target_site]["predicted"])
#             predonebpdelfreq.append(predonebpdelratio)
#             obsonebpdelratio = sum(np.array(data[t][method][target_site]["actual"])[onebpdels])/sum(data[t][method][target_site]["actual"])
#             obsonebpdelfreq.append(obsonebpdelratio)
#         onebpdelmse = mean_squared_error(predonebpdelfreq, obsonebpdelfreq)


#         # one basepair insertions
#         predonebpinsfreq = []
#         obsonebpinsfreq = []
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"]) 
#             onebpins = [is_1bp_insertion(x, method) for x in indels]
#             predonebpinsratio = sum(np.array(data[t][method][target_site]["predicted"])[onebpins])/sum(data[t][method][target_site]["predicted"])
#             predonebpinsfreq.append(predonebpinsratio)
#             obsonebpinsratio = sum(np.array(data[t][method][target_site]["actual"])[onebpins])/sum(data[t][method][target_site]["actual"])
#             obsonebpinsfreq.append(obsonebpinsratio)
#         onebpinsmse = mean_squared_error(predonebpinsfreq, obsonebpinsfreq)

#         # one bp frameshift
#         predonebpframeshiftfreq = []
#         obsonebpframeshiftfreq = []
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"]) 
#             onebpframeshift = [is_frameshift(x, method, 1) for x in indels]
#             predonebpframeshiftratio = sum(np.array(data[t][method][target_site]["predicted"])[onebpframeshift])/sum(data[t][method][target_site]["predicted"])
#             predonebpframeshiftfreq.append(predonebpframeshiftratio)
#             obsonebpframeshiftratio = sum(np.array(data[t][method][target_site]["actual"])[onebpframeshift])/sum(data[t][method][target_site]["actual"])
#             obsonebpframeshiftfreq.append(obsonebpframeshiftratio)
#         onebpframeshiftmse = mean_squared_error(predonebpframeshiftfreq, obsonebpframeshiftfreq)

#         # two bp frameshift
#         predtwobpframeshiftfreq = []
#         obstwobpframeshiftfreq = []
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"]) 
#             twobpframeshift = [is_frameshift(x, method, 2) for x in indels]
#             predtwobpframeshiftratio = sum(np.array(data[t][method][target_site]["predicted"])[twobpframeshift])/sum(data[t][method][target_site]["predicted"])
#             predtwobpframeshiftfreq.append(predtwobpframeshiftratio)
#             obstwobpframeshiftratio = sum(np.array(data[t][method][target_site]["actual"])[twobpframeshift])/sum(data[t][method][target_site]["actual"])
#             obstwobpframeshiftfreq.append(obstwobpframeshiftratio)
#         twobpframeshiftmse = mean_squared_error(predtwobpframeshiftfreq, obstwobpframeshiftfreq)
        
#         # frameshift
#         predframeshiftfreq = []
#         obsframeshiftfreq = []
#         for target_site in common_oligos[t]:
#             indels = np.array(data[t][method][target_site]["indels"]) 
#             frameshift = [is_frameshift(x, method, "any") for x in indels]
#             predframeshiftratio = sum(np.array(data[t][method][target_site]["predicted"])[frameshift])/sum(data[t][method][target_site]["predicted"])
#             predframeshiftfreq.append(predframeshiftratio)
#             obsframeshiftratio = sum(np.array(data[t][method][target_site]["actual"])[frameshift])/sum(data[t][method][target_site]["actual"])
#             obsframeshiftfreq.append(obsframeshiftratio)
#         frameshiftmse = mean_squared_error(predframeshiftfreq, obsframeshiftfreq)


#         rows.append([delmse, onebpdelmse, onebpinsmse, onebpframeshiftmse, twobpframeshiftmse, frameshiftmse])
#         indices.append((test_f, method))
  
# indices = pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method"])
# df = pd.DataFrame(rows, index=indices, columns=["Deletion", "1BP Deletion", "1BP Insertion", "1BP Frameshift", "2BP Frameshift", "Frameshift"])
# df.groupby(["Dataset", "Method"])


CROTON_F = "/Users/colm/repos/output/local/model_predictions/CROTON/{}_new.pkl"

summ_data = {}

for i, t in enumerate(TEST_FILES):
    croton_d = pkl.load(open(CROTON_F.format(t), 'rb'))
    summ_data[t] = croton_d

croton_i = pd.MultiIndex.from_product([[file_mapping[t] for t in TEST_FILES], ["CROTON"]], names=["Dataset", "Method"])


croton_del_freq_mse = []
croton_prob_1bpins_mse = []
croton_prob_1bpdel_mse = []
croton_one_bp_frameshift_mse = []
croton_two_bp_frameshift_mse = []
croton_frameshift_mse = []

for t in TEST_FILES:
    croton_del_freq_mse.append(mean_squared_error(summ_data[t]["predicted"]["del_freq"], summ_data[t]["actual"]["del_freq"]))
    croton_prob_1bpins_mse.append(mean_squared_error(summ_data[t]["predicted"]["prob_1bpins"], summ_data[t]["actual"]["prob_1bpins"]))
    croton_prob_1bpdel_mse.append(mean_squared_error(summ_data[t]["predicted"]["prob_1bpdel"], summ_data[t]["actual"]["prob_1bpdel"]))
    croton_one_bp_frameshift_mse.append(mean_squared_error(summ_data[t]["predicted"]["one_bp_frameshift"], summ_data[t]["actual"]["one_bp_frameshift"]))
    croton_two_bp_frameshift_mse.append(mean_squared_error(summ_data[t]["predicted"]["two_bp_frameshift"], summ_data[t]["actual"]["two_bp_frameshift"]))
    croton_frameshift_mse.append(mean_squared_error(summ_data[t]["predicted"]["frameshift"], summ_data[t]["actual"]["frameshift"]))

croton_mse_d = pd.DataFrame({
    "Deletion": croton_del_freq_mse,
    "1BP Insertion": croton_prob_1bpins_mse,
    "1BP Deletion": croton_prob_1bpdel_mse,
    "1BP Frameshift": croton_one_bp_frameshift_mse,
    "2BP Frameshift": croton_two_bp_frameshift_mse,
    "Frameshift": croton_frameshift_mse,
}, index=croton_i)

croton_mse_d

df = pd.concat([df, croton_mse_d.reset_index()]).sort_index()
df.to_csv(os.environ["OUTPUT_DIR"] + "/Results/Transfer_Learning/stats_comparison.tsv", sep="\t")

print("Calculated stats results")

