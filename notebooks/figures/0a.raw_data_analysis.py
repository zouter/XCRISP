# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import percentile

# %%
all_ds = ['FORECasT',
 '0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1',
 '1027-mESC-Lib1-Cas9-Tol2-Biorep1-techrep1',
 '052218-U2OS-+-LibA-postCas9-rep1',
 '0226-PRLmESC-Lib1-Cas9',
 'TREX_A',
 'HAP1']

title_mapping = {
    'FORECasT': "FORECasT mESC WT",
    '0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1': "inDelphi mESC WT",
    '1027-mESC-Lib1-Cas9-Tol2-Biorep1-techrep1': "inDelphi mESC WT",
    "052218-U2OS-+-LibA-postCas9-rep1": "U2OS",
    '0226-PRLmESC-Lib1-Cas9': 'mESC ($\it{\mathregular{NHEJ^{-/-}}}$)',
    'TREX_A': "TREX2",
    'HAP1': "mESC HAP1"
}

# for d in all_ds:

#     file_dir = os.environ["OUTPUT_DIR"] + "processed_data/Tijsterman_Analyser/{}/".format(d)

#     if not os.path.exists(file_dir):
#         print(file_dir, "does not exist.")
#         continue

#     all_f = os.listdir(file_dir)
#     all_f = [f for f in all_f if ("tij.sorted" in f)]

#     if len(all_f) == 0:
#         continue

#     df = []

#     for f in all_f:
#         target = f.split(".")[0]
#         counts = pd.read_csv(file_dir + f, sep="\t")["countEvents"].sum()
#         df.append((target, counts))

#     df = pd.DataFrame(df, columns=["Target", "Count"])
#     df.head()

#     data = df["Count"]
#     quartiles = percentile(data, [25, 50, 75])
#     data_min, data_max = data.min(), data.max()

#     print(d)
#     print(f"Min: {data_min:.3f}")
#     print('Q1: %.3f' % quartiles[0])
#     print('Median: %.3f' % quartiles[1])
#     print('Q3: %.3f' % quartiles[2])
#     print('Max: %.3f' % data_max)

#     p = sns.displot(data=df, x="Count", aspect=4, height=3, binwidth=100 if d in ["FORECasT", "TREX_A", "HAP1"] else 10000)
#     p.figure.suptitle(f"Distribution of mutated read counts for {title_mapping[d]}\n" + 
#         f"Min: {data_min:.3f} \n" + 
#         f"Q1: {quartiles[0]:.3f} \n" + 
#         f"Median: {quartiles[1]:.3f} \n" + 
#         f"Q3: {quartiles[2]:.3f} \n" + 
#         f"Max: {data_max:.3f} \n"       
#     )
#     # plt.show()
#     plt.savefig(f"./notebooks/figures/artifacts/mutated_read_count_distributions_{d}.pdf")
# plt.close()


for d in all_ds:

    file_dir = os.environ["OUTPUT_DIR"] + "processed_data/Tijsterman_Analyser/{}/".format(d)

    if not os.path.exists(file_dir):
        continue

    all_f = os.listdir(file_dir)
    all_f = [f for f in all_f if ("tij.sorted" in f)]

    if len(all_f) == 0:
        continue

    df = []

    for f in all_f:
        target = f.split(".")[0]
        counts = pd.read_csv(file_dir + f, sep="\t")["countEvents"].sum()
        t = pd.read_csv(file_dir + f, sep="\t")
        total_counts = t["countEvents"].sum()
        t["Fraction"] = t["countEvents"]/total_counts
        t = t[t["Type"] == "DELETION"][["Type", "Size", "Fraction"]].groupby(["Type", "Size"]).sum(numeric_only=True)
        df.append(t)

    df = pd.concat(df).reset_index()
    fig, ax = plt.subplots(figsize=(15, 3))
    sns.barplot(data=df.groupby("Size").sum(numeric_only=True).div(len(all_f)).reset_index(), x="Size", y="Fraction", ax=ax)
    plt.title(f"Mean Deletion Length Frequency of Mapped Mutated Reads for\n{title_mapping[d]}")
    plt.ylim(0, 0.2)
    plt.tight_layout()
    plt.savefig(f"./notebooks/figures/artifacts/mapped_mutated_reads_deletion_length_frequencies_{d}.pdf", bbox_inches='tight')
plt.close()


for d in all_ds:

    file_dir = os.environ["OUTPUT_DIR"] + "processed_data/Tijsterman_Analyser/{}/".format(d)

    if not os.path.exists(file_dir):
        continue

    all_f = os.listdir(file_dir)
    all_f = [f for f in all_f if ("tij.sorted" in f)]

    if len(all_f) == 0:
        continue

    df = []

    for f in all_f:
        target = f.split(".")[0]
        counts = pd.read_csv(file_dir + f, sep="\t")["countEvents"].sum()
        t = pd.read_csv(file_dir + f, sep="\t")
        total_counts = t["countEvents"].sum()
        t["Fraction"] = t["countEvents"]/total_counts
        t = t[t["Type"] == "INSERTION"][["Type", "Size", "Fraction"]].groupby(["Type", "Size"]).sum(numeric_only=True)
        df.append(t)

    df = pd.concat(df).reset_index()
    fig, ax = plt.subplots(figsize=(15, 3))
    sns.barplot(data=df.groupby("Size").sum(numeric_only=True).div(len(all_f)).reset_index(), x="Size", y="Fraction", ax=ax)
    plt.title(f"Mean Insertion Length Frequency of Mapped Mutated Reads for\n{title_mapping[d]}")
    plt.ylim(0, 0.4)
    plt.tight_layout()
    plt.savefig(f"./notebooks/figures/artifacts/mapped_mutated_reads_insertion_length_frequencies_{d}.pdf", bbox_inches='tight')
plt.close()
