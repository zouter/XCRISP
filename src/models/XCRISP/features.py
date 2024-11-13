import numpy as np
import pandas as pd
from Bio.SeqUtils import GC
from scipy.stats import rankdata
from src.models.XCRISP.indels import gen_indels_v3

BASE_FEATURES = ["Size", "Start", "leftEdge", "leftEdgePositive", "rightEdge", "numRepeats", "homologyLength", "leftEdgeMostDownstream", "rightEdgeMostUpstream"]
EXTRA_HOMOLOGY_FEATURES = ["homologyGCContent", "homologyDistanceRank", "homologyLeftEdgeRank", "homologyRightEdgeRank", "homologyLengthRank", "homologyNearPosition", "homologyFarPosition", "homologyGap"]
DELETION_FEATURES = BASE_FEATURES + EXTRA_HOMOLOGY_FEATURES

def onehotencoder(seq):
    '''convert to single and di-nucleotide hotencode'''
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    encode_2d = encode.reshape((96, 4))
    return encode, head_idx, encode_2d

def get_closest_edge_to_cutsite(positions):
    idx = np.argmin(np.abs(positions))
    return positions[idx]

def get_most_upstream_edge(positions):
    idx = np.argmin(positions)
    return positions[idx]

def get_most_downstream_edge(positions):
    idx = np.argmax(positions)
    return positions[idx]

def get_features(indels):
    indels["leftEdge"] = indels["Positions"].apply(lambda x: get_closest_edge_to_cutsite(x)) # left edge closest to cutsite
    indels["leftEdgeMostDownstream"] = indels["Positions"].apply(lambda x: get_most_downstream_edge(x)) # left edge closest to cutsite
    indels["rightEdge"] = indels.apply(lambda x: get_closest_edge_to_cutsite(np.array(x["Positions"]) + x["Size"]), axis=1) # right edge closest to cutsite
    indels["rightEdgeMostUpstream"] = indels.apply(lambda x: get_most_upstream_edge(np.array(x["Positions"]) + x["Size"]), axis=1) # right edge closest to cutsite
    indels["leftEdgePositive"] = -indels["leftEdge"].copy()

    # microhomology based features
    mh_indels = indels[indels["homologyLength"] > 0].copy()
    mh_indels["homologyGCContent"] = mh_indels["homology"].apply(lambda x: GC(x)/100)
    mh_indels["homologyLeftEdgeRank"] = rankdata(np.abs(mh_indels["leftEdge"]), method="dense") # left edge closest to cutsite, ranked
    mh_indels["homologyRightEdgeRank"] = rankdata(np.abs(mh_indels["rightEdge"]), method="dense") # right edge closest to cutsite, ranked
    mh_indels["homologyDistanceRank"] = rankdata(mh_indels["Size"] - mh_indels["homologyLength"], method="dense") # closest MH
    mh_indels["homologyLengthRank"] = rankdata(mh_indels["homologyLength"]*-1, method="dense") # longest homology
    mh_indels["homologyNearPosition"] = mh_indels[["leftEdgePositive", "rightEdge"]].min(axis=1)
    mh_indels["homologyFarPosition"] = mh_indels[["leftEdgePositive", "rightEdge"]].max(axis=1)
    mh_indels["homologyGap"] = mh_indels["rightEdge"] - mh_indels["leftEdge"]


    indels = indels.join(mh_indels[EXTRA_HOMOLOGY_FEATURES])
    indels[EXTRA_HOMOLOGY_FEATURES] = indels[EXTRA_HOMOLOGY_FEATURES].fillna(0)
    return indels[DELETION_FEATURES]

def get_insertion_features(seq, cutsite):
    guide = seq[cutsite-17:cutsite+3]
    encoding, index, encoding_2d = onehotencoder(guide)
    encoding = pd.Series(encoding, index=index)
    return encoding

if __name__ == "__main__":
    # seq = "GTGCTCTTAACTTTCACTTTATAGATTTATAGGGTTAATAAATGGGAATTTATAT"
    seq = "TCACTACAAGTCAGGAATGCCTGCGTTTGGCCGTCCAGTTAGTAACAGAAGGTCAGGTAAGAGG"
    cutsite = 27
    indels = gen_indels_v3(seq, cutsite, max_deletion_length=10)
    features = get_features(indels)
    print(features[["homologyLength"]])
    # print(features[(features["homologyGCContent"] != 0) & (features["homologyGCContent"] != 1)])
    # insertion_features = get_insertion_features(seq, cutsite)
    # print(insertion_features)
    print("done.")

