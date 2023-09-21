import pandas as pd
# run file on data from indelphi-dataprocessinganalysis/data-libprocessing

def read_file(fname):
    return [g.strip() for g in open(fname, "r").readlines()]

def get_data(lib):
    data_dir = "indelphi-dataprocessinganalysis/data-libprocessing/"

    grnas = read_file(data_dir + "grna-lib{}.txt".format(lib))
    names = read_file(data_dir + "names-lib{}.txt".format(lib))
    targets = read_file(data_dir + "targets-lib{}.txt".format(lib))
    df = pd.DataFrame({
        "Name": names,
        "Guide": grnas,
        "Full target sequence": targets
    })
    return df

if __name__ == "__main__":
    for lib in ["A", "B"]:
        d = get_data(lib)
        d.to_csv("Lib{}.csv".format(lib), index=False)