from math import modf
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import datetime
j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

def get_julian_datetime(date):
    if date.year < 1801 or date.year > 2099:
        raise ValueError('Datetime must be between year 1801 and 2099')

    julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int((275 * date.month) / 9.0) + date.day + 1721013.5 + (date.hour + date.minute / 60.0 + date.second / np.power(60,2)) / 24.0 - 0.5 * np.copysign(1, 100 * date.year + date.month - 190002.5) + 0.5

    return julian_datetime

def get_gregorian_datetime(date):
    date = float(date)
    date_f, date_i = np.modf(date)
    date_f = float(date_f)
    date_i = int(date_i)
    if -0.5 < date_f < 0.5:
        date_f += 0.5
    elif date_f >= 0.5:
        date_i += 1
        date_f -= 0.5
    elif date_f <= -0.5:
        date_i -= 1
        date_f += 1.5

    ell = date_i + 68569
    n = int((4 * ell) / 146097.0)
    ell -= int(((146097 * n) + 3) / 4.0)
    i = int((4000 * (ell + 1)) / 1461001)
    ell -= int((1461 * i) / 4.0) - 31
    j = int((80 * ell) / 2447.0)
    day = ell - int((2447 * j) / 80.0)
    ell = int(j / 11.0)
    month = j + 2 - (12 * ell)
    year = 100 * (n - 49) + i + ell
    date_f, hour = np.modf(24 * date_f)
    date_f, minute = np.modf(60 * date_f)
    date_f, second = np.modf(60 * date_f)
    date_f, microsecond = np.modf(1000000 * date_f)
    return datetime.datetime(year=year, month=month, day=day, hour=int(hour), minute=int(minute), second=int(second), microsecond=int(microsecond))

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


rawdata = np.genfromtxt('./Tres-1b/Tres-1b_Mag_diff.txt', delimiter=' ', skip_header=2)
# Weird data in line 120 because Muniwin did not cross reference the variable star
mask = np.ones(len(rawdata), dtype=bool)
mask[[117]] = False
rawdata = rawdata[mask,...]
data = rawdata[:, :3]
t = data[:, 0]
VC = data[:, 1]
VCmax, VCmin, VCmean = VC.max(), VC.min(), VC.mean()
VCerr = data[:, 2]
plt.plot(t, VC)
plt.ylim(VCmax + 0.05 * VCmean, VCmin - 0.05 * VCmean)
plt.show()

f = make_piecewise([[0.5, 10, 20], [-0.5, 0.5, 1]])
x = np.linspace(0, 2, 10)
plt.plot(x, f(x))
plt.show()