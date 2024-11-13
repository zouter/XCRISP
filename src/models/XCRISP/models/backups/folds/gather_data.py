import pandas as pd

COVERAGES = [1000]
RANDOM_STATE = [1, 42, 919]
LOSSES = ["Base"]
EXPERIMENTS = ["full", "full+numrepeat", "inDelphi_features", "v2", "v2+ranked"]
OUTPUTS = ["Sigmoid", "Exp"]

FILE = "{}_{}x_{}_{}Loss_RS_{}_model..folds.tsv"

dfs = []

for o in OUTPUTS:
    for c in COVERAGES:
        for e in EXPERIMENTS: 
            if (o == "Exp") and e in ["v2", "v2+ranked"]: continue
            for l in LOSSES:
                for r in RANDOM_STATE:
                    data = pd.read_csv(FILE.format(o, c, e, l, r), sep="\t")
                    data.columns = ["Fold", "Loss", "Train Corr", "Val Corr"]
                    data = data.head()
                    data["Random_State"] = r
                    data["Loss"] = l
                    data["Experiment"] = e
                    dfs.append(data)

    df = pd.concat(dfs)
    df = df.set_index(["Experiment", "Loss", "Random_State"])
    df.to_csv("CV_OurModel_{}x_{}.tsv".format(c, o), sep="\t")
