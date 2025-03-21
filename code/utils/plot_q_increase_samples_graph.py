import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


# http://www.ColorBrewer.org - for color blind friendly graphs
colour_scheme = [
['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e'],
['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99'],
['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'],
['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854'],
['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3']
]


def plot(q, x, filename, choice = 0):

    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=5)
    plt.rc('legend', fontsize=8)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(direction="in")
    plt.rcParams.update({'font.size': 32})
    
    # plot grid
    plt.grid(alpha=0.2, linestyle='dotted', c="black")

    # set ticks
    ax.set_xticks(x, minor=False)
    tick_spacing = 20
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    # increase the limits of the fig to see the end points clearly
    ax.set_xlim(min(x), max(x)+2)
    
    ax.plot(x, q, color="c", linewidth=1.5, linestyle="dashed")
    ax.set_xlabel("Percentage of data used for variational training", fontsize=16)
    ax.set_ylabel("Prop@10", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    fig.savefig(filename, bbox_inches='tight')


def main():
    dir_name = "plots"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(dir_name, "q_increase_samples_graph.pdf")

    q = np.asarray([64.8148148148148, 65.2777777777777, 68.0555555555555, 68.287037037037, 66.2037037, 68.75])
    x = np.asarray([0, 20, 40, 60, 80, 100], dtype=int)

    plot(q, x, filename)


if __name__ == "__main__":
    main()