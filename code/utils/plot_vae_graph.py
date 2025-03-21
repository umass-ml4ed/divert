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


def plot(vae, e_d_given_s, e_given_s_plus_d_given_s_e, x, filename, choice = 0):

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
    
    ax.plot(x, vae, color="c", linewidth=1.5, linestyle="dashed", label="DiVERT")
    ax.plot(x, e_d_given_s, color="m", linewidth=1.5, linestyle="dashed", label="DisSearch-ED COT")
    ax.plot(x, e_given_s_plus_d_given_s_e, color="y", linewidth=1.5, linestyle="dashed", label="DisSearch-ED COT Pipeline")
    
    ax.legend(loc='lower left', frameon=True, framealpha=1, edgecolor="black", fontsize=14)
    ax.set_xlabel("Percentage of error labels dropped", fontsize=16)
    ax.set_ylabel("Prop@10", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    fig.savefig(filename, bbox_inches='tight')


def main():
    dir_name = "plots"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(dir_name, "vae_graph.pdf")

    vae = np.asarray([68.75, 67.3611111111111, 61.3425925925925, 63.6574074074074, 57.6388888888888])
    e_d_given_s = np.asarray([66.6666666666666, 66.6666666666666, 61.1111111111111, 56.9444444444444, 50.6944444444444])
    e_given_s_plus_d_given_s_e = np.asarray([64.5833333333333, 65.97222222, 59.02777778, 55.78703704, 53.00925926])
    x = np.asarray([0, 20, 40, 60, 80], dtype=int)

    plot(vae, e_d_given_s, e_given_s_plus_d_given_s_e, x, filename)


if __name__ == "__main__":
    main()