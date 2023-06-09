import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

""" This functions creates graphs to compare all algorithms & vectorization methods. 
    The graphs are available at the final report"""

def autolabel(rects, labels, ax):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                str("%.3f" % labels[idx]),
                ha='center', va='bottom', rotation=0, fontsize='medium')


def plot_bars(classifiers, scores, metric):
    fig, ax = plt.subplots(figsize=(13.2, 8), layout='constrained')
    ax.set_title(metric)
    ax.set_ylabel("score")
    bar_plot = ax.bar(classifiers, scores)
    ax.set_xticks(range(len(classifiers)), classifiers, rotation=90, fontsize='small')
    autolabel(bar_plot, scores, ax)
    fig.savefig("{}.png".format(metric))


def plot_loss_graphs(classifiers, losses):
    fig, ax = plt.subplots(figsize=(13.2, 8))
    x = np.linspace(0, 1, len(losses[0]))
    for i in range(len(classifiers)):
        ax.plot(x, losses[i], label=classifiers[i])
    ax.set_title("Train Loss")
    ax.legend(fontsize='xx-small', loc=1)
    fig.savefig("train_loss.png")


def plot_rocs(classifiers, rocs):
    fig, ax = plt.subplots(figsize=(13.2, 8))
    for i in range(len(classifiers)):
        ax.plot(rocs['fpr'][i], rocs['tpr'][i], label=classifiers[i])
    ax.set_title("Roc curves")
    ax.set_xlabel("Fpr")
    ax.set_ylabel("Tpr")
    ax.legend(fontsize='small', loc=4)
    fig.savefig("roc_curves.png")


def plot_all(data):
    """
    The method creates graphs to compare all algorithms & vectorization methods
    Args:
        data:

    Returns:

    """
    losses = []
    losses_classifiers = []
    classifiers = []
    bars = {'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': []}
    rocs = {'fpr': [], 'tpr': []}

    ## collect data
    for c_data in data:
        cls_name = "{}_{}".format(c_data["classifier"].to_string(), c_data["vectorize"].to_string())
        classifiers.append(cls_name)
        if (hasattr(c_data["classifier"], 'losses')):
            losses.append(list(map(lambda x: x.detach().numpy(), c_data["classifier"].losses)))
            losses_classifiers.append(cls_name)
        for metric in bars.keys():
            bars[metric].append(c_data["scores"][metric])
        rocs['fpr'].append(c_data["fpr"])
        rocs['tpr'].append(c_data["tpr"])

    ## plot bars data
    for metric in bars.keys():
        plot_bars(classifiers, bars[metric], metric)

    ## plot rocs
    plot_rocs(classifiers, rocs)
