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
        df.append((target, counts))

    df = pd.DataFrame(df, columns=["Target", "Count"])
    df.head()

    data = df["Count"]
    quartiles = percentile(data, [25, 50, 75])
    data_min, data_max = data.min(), data.max()

    print(d)
    print(f"Min: {data_min:.3f}")
    print('Q1: %.3f' % quartiles[0])
    print('Median: %.3f' % quartiles[1])
    print('Q3: %.3f' % quartiles[2])
    print('Max: %.3f' % data_max)

    sns.displot(data=df, x="Count", aspect=4, height=3, binwidth=100 if d in ["FORECasT", "TREX_A", "HAP1"] else 10000)
    plt.title(f"Distribution of mutated read counts for {title_mapping[d]}\n" + 
        f"Min: {data_min:.3f} \n" + 
        f"Q1: {quartiles[0]:.3f} \n" + 
        f"Median: {quartiles[1]:.3f} \n" + 
        f"Q3: {quartiles[2]:.3f} \n" + 
        f"Max: {data_max:.3f} \n"       
    )
    plt.show()
    # plt.savefig(f"./artifacts/mutated_read_count_distributions_{d}.pdf")
plt.close()


