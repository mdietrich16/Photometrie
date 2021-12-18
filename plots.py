import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    rawdata = np.genfromtxt(filename, delimiter=' ', skip_header=2)
    t = rawdata[:, 0]
    data = rawdata[:, 1::2]
    errs = rawdata[:, 2::2]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    mask = np.abs(data - mean) / std > 5
    if np.any(mask):
        mask = np.any(mask, axis=1).nonzero()[0]
        t = np.delete(t, mask, 0)
        errs = np.delete(errs, mask, 0)
        data = np.delete(data, mask, 0)
    t -= t[0]
    data = np.power(10, -data/2.5)
    errs = np.log(10)/2.5 * np.power(10, -data/2.5) * errs
    return t, data, errs


def plot_data(t, data, errs, n, k, name):

    # V vs k
    kwargs = {"xlabel": "Zeit in Tagen"}
    figVK, axVK = plt.subplots(
        nrows=k, ncols=1, subplot_kw=kwargs, layout='constrained', figsize=(14, 10), dpi=200, sharex='col')
    figVK.suptitle("Strahlungsflussverhältnisse V vs K")
    for i in range(k):
        axVK[i].errorbar(x=t, y=data[:, 1+i], yerr=errs[:, 1+i],
                         fmt='.', markersize=4, elinewidth=.8, capsize=1)
        axVK[i].set_ylabel("$\\frac{S_V}{S_{K" + str(i+1) + "}}$", fontsize=20)
        axVK[i].grid(True, color='gray', linewidth=0.1)

    figVK.savefig(name + "_VK.png", dpi=200, bbox_inches='tight')

    # C vs k
    figCK, axCK = plt.subplots(
        n, k, subplot_kw=kwargs, layout='constrained', figsize=(14, 10), dpi=200, sharex='col')
    figCK.suptitle("Strahlungsflussverhältnisse C vs K")
    for i in range(n):
        for j in range(k):
            axCK[i, j].errorbar(
                t, data[:, 1+k+i*k+j], errs[:, 1+k+i*k+j], fmt='.', markersize=4, elinewidth=.8, capsize=1)
            axCK[i, j].set_ylabel(
                "$\\frac{S_{C" + str(i+1) + "}}{S_{K" + str(j+1) + "}}$", fontsize=20)
            axCK[i, j].grid(True, color='gray', linewidth=0.1)

    figCK.savefig(name + "_CK.png", dpi=200, bbox_inches='tight')

    # K vs K
    figKK, axKK = plt.subplots(
        k*(k-1)//2, 1, subplot_kw=kwargs, layout='constrained', figsize=(14, 10), dpi=200, sharex='col')
    figKK.suptitle("Strahlungsflussverhältnisse K vs K")
    for i in range(k*(k-1)//2):
        axKK[i].errorbar(t, data[:, 1+k+k*n+i], errs[:, 1+k +
                         k*n+i], fmt='.', markersize=4, elinewidth=.8, capsize=1)
        x = int(0.5 + np.sqrt(0.25 + 2*i))
        y = i - x*(x-1)//2
        axKK[i].set_ylabel(
            "$\\frac{S_{K" + str(x + 1) + "}}{S_{K" + str(y + 1) + "}}$", fontsize=20)
        axKK[i].grid(True, color='gray', linewidth=0.1)

    figKK.savefig(name + "_KK.png", dpi=200, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    t, data, errs = load_data(
        'UCAC4_558-007313/UCAC4_558-007313_Light_curve.txt')
    plot_data(t, data, errs, 3, 3, 'UCAC')
    t, data, errs = load_data(
        './Tres-1b/Tres-1b-Sel.txt')
    plot_data(t, data, errs, 5, 4, 'Tres-1b')
