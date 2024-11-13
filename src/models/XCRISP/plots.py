import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np

def get_color_for_indel(indel, shade="dark"):
    colors = cm.get_cmap("tab20").colors
    dark_blue = (0,0,0)
    light_blue = colors[1]
    dark_red = (1,0,0)
    light_red = colors[13]
    if shade=="dark":
        return dark_red if indel[0] == "I" else dark_blue
    else:
        return light_red if indel[0] == "I" else light_blue

def plot_indel_distribution(target, indels, counts, observ_counts):
    fig = plt.figure(figsize=(30, 5), dpi=120, facecolor='w', edgecolor='k')

    # set plot values
    fontsize = 4
    bar_width = 0.5
    dark_colors = [get_color_for_indel(x, "dark") for x in indels]
    light_colors = [get_color_for_indel(x, "light") for x in indels]

    # plot predicted
    y_pos = np.arange(len(indels))
    nc = np.array(counts)
    # nc = nc/nc.sum()
    plt.bar(y_pos, nc, color=dark_colors, width=bar_width)

    if observ_counts is not None:
        # plot observed
        nobsc = np.array(observ_counts)
        # nobsc = nobsc/nobsc.sum()
        y_pos_2 = [x + bar_width for x in y_pos]
        plt.bar(y_pos_2, nobsc, color=light_colors, width=bar_width)
        corr = np.corrcoef(counts, observ_counts)[0,1]

    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.xticks(y_pos, indels, fontsize=fontsize, rotation=90)
    plt.ylabel('# Reads')
    if observ_counts is not None:
        plt.title("{0},\n Corr: {1:.2f}, Reads: {2}".format(target, corr, sum(observ_counts)))
    else:
        plt.title("{0},\nReads: {1}".format(target, sum(counts)))
    return fig
