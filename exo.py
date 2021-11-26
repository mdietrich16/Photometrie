from math import modf
import numpy as np
from numpy import vectorize
import scipy.optimize as opt
import matplotlib.pyplot as plt

import datetime

@vectorize
def jul_to_greg(jd):
    assert np.all(jd >= 0)
    frac, jd = np.modf(jd)
    f = int(jd + 1401 + (((4 * jd + 274277)//146097) * 3) // 4 - 38)
    e = 4 * f + 3
    g = (e % 1461) // 4
    h = 5 * g + 2
    day = (h % 153) // 5 + 1
    month = (h // 153 + 2) % 12 + 1
    year = (e // 1461) - 4716 + (14 - month) // 12
    frac, hour = np.modf(24 * frac)
    frac, minute = np.modf(60 * frac)
    frac, second = np.modf(60 * frac)
    frac, microsecond = np.modf(1000000 * frac)
    return datetime.datetime(year=year, month=month, day=day, hour=12+int(hour), minute=int(minute), second=int(second), microsecond=int(microsecond))

def greg_to_jul(dt):
    # jdn = (1461 * (dt.year + 4800 + (dt.month - 14) // 12)) // 4 + (367 * (dt.month - 2 - 12 * ((dt.month - 14) // 12))) // 12 - (3 * ((dt.year + 4900 + (dt.month - 14) // 12) // 100)) // 4 + dt.day - 32075
    # return float(jdn) + float(dt.hour - 12)/24 + float(dt.minute)/1440 + float(dt.second)/86400 + float(dt.microsecond)/1000000
    jd = 367 * dt.year - int((7 * (dt.year + int((dt.month + 9) / 12.0))) / 4.0) + int((275 * dt.month) / 9.0) + dt.day + 1721013.5 +(dt.hour + dt.minute / 60.0 + dt.second / np.power(60,2) + dt.microsecond / (np.power(60,2) * 1000000)) / 24.0 - 0.5 * np.copysign(1, 100 * dt.year + dt.month - 190002.5) + 0.5
    return jd

def make_piecewise(params):
    params = np.array(params)
    assert len(params.shape) == 2
    assert params.shape[1] == 3
    n = params.shape[0]

    funcs = []
    boundaries = []
    for m,b,x_min in params:
        print(m, b, x_min)
        funcs.append(lambda x: m*x + b)
        boundaries.append(lambda x: x >= x_min)
    #return lambda x: np.piecewise(x, [cond(x) for cond in reversed(boundaries)], list(reversed(funcs)))
    print(boundaries)
    def f(x):
        conds = [cond(x) for cond in boundaries]
        print(conds)
        print(funcs)
        return np.piecewise(x, conds, funcs)
    return f

def load_exo_data():
    rawdata = np.genfromtxt('./Tres-1b/Tres-1b_Mag_diff.txt', delimiter=' ', skip_header=2)
    # Weird data in line 120 because Muniwin did not cross reference the variable star
    mask = np.ones(len(rawdata), dtype=bool)
    mask[[117]] = False
    rawdata = rawdata[mask,...]
    return rawdata

def process_exo_data(rawdata):
    data = rawdata[:, :3]
    t = data[:, 0]
    VC = data[:, 1]
    VCmax, VCmin, VCmean = VC.max(), VC.min(), VC.mean()
    VCerr = data[:, 2]
    return t, VC, VCerr, VCmax, VCmin, VCmean

def plot_exo_data(data):
    t, VC, VCerr, VCmax, VCmin, VCmean = data
    plt.plot(t, VC)
    plt.ylim(VCmax + 0.05 * VCmean, VCmin - 0.05 * VCmean)
    plt.show()

if __name__ == "__main__":
    #f = lambda x,m,b,n,c: (x > 0 ? (x > 1 ? m*x + b : (x > 2 ? 0 : n*x + c)) : 0)
    #x = np.linspace(0, 2, 10)
    #plt.plot(x, f(x))
    #plt.show()
    rawdata = load_exo_data()
    data = process_exo_data(rawdata)
    print(jul_to_greg(data[0]))
