from math import modf
import numpy as np
from numpy import vectorize
from numpy.core.fromnumeric import var
from numpy.linalg import cond
import scipy.optimize as opt
from scipy.special import factorial
import matplotlib.pyplot as plt

import datetime
jdt_exo = 2.459492e6
jdt_var = 2.459543e6


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
    jd = 367 * dt.year - int((7 * (dt.year + int((dt.month + 9) / 12.0))) / 4.0) + int((275 * dt.month) / 9.0) + dt.day + 1721013.5 + (dt.hour + dt.minute /
                                                                                                                                       60.0 + dt.second / np.power(60, 2) + dt.microsecond / (np.power(60, 2) * 1000000)) / 24.0 - 0.5 * np.copysign(1, 100 * dt.year + dt.month - 190002.5) + 0.5
    return jd


def trapez(x, p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y):
    condlist = [np.logical_or(x < p0x, x > p3x), np.logical_and(
        x < p1x, x > p0x), np.logical_and(x < p2x, x > p1x), np.logical_and(x < p3x, x > p2x)]

    def steady(x):
        return (p3y - p0y) / (p3x - p0x) * (x - p0x) + p0y

    def trans_in(x):
        return (p1y - p0y) / (p1x - p0x) * (x - p0x) + p0y

    def trans_total(x):
        return (p2y - p1y) / (p2x - p1x) * (x - p1x) + p1y

    def trans_out(x):
        return (p3y - p2y) / (p3x - p2x) * (x - p2x) + p2y

    return np.piecewise(x, condlist=condlist, funclist=[steady, trans_in, trans_total, trans_out])


def transit(x, m, b, tm, tt, td, ttd):
    # x: Input
    # m: Steigung der Hauptgeraden
    # b: y-Achsen-Abschnitt der Hauptgeraden
    # tm: Transitmitte
    # tt: Transittiefe
    # td: Transitdauer
    # ttd: Totalitätsdauer
    y = m * x + b
    td2, ttd2 = td/2, ttd/2
    diff = td2 - ttd2
    slope = tt / diff
    condlist = [np.logical_and(x > tm-td/2, x < tm-ttd/2), np.logical_and(
        x > tm-ttd/2, x < tm+ttd/2), np.logical_and(x > tm+ttd/2, x < tm+td/2)]
    y += np.piecewise(x, condlist, funclist=[lambda x: -slope * (
        x - (tm - td2)), lambda x: -tt, lambda x: slope * (x - (tm + td2)), lambda x: 0])
    return y


def lerp(t, p1, p2):
    return (1 - t)*p1 + t*p2


def nCk(n, k):
    return factorial(n)/(factorial(n-k)*factorial(k))


def bezier(t, points):
    t = t.reshape(-1, 1)
    degree = points.shape[0]-1
    i = np.tile(np.arange(degree+1), (t.shape[0], 1))
    return np.dot(nCk(degree, i)*np.power(1 - t, degree - i)*np.power(t, i), points)

# def bezier(t, points):
#    num = len(points)
#    if num == 1:
#        return points[0]
#    else:
#        return lerp(t, bezier(t, points[:-1]), bezier(t, points[1:]))


def transit_bezier(x, x0, xn, *points):
    dx = 0.05
    points = np.array(points)
    x = np.array(x)
    dx2 = xn - x0
    # return m * x + b + (x > x0)*(x < xn)*bezier((x-x0)/dx, points)[:,1]
    # return bezier((x-x0)/dx, points)

    def linear(x):
        return (points[-1] - points[0])/dx2 * (x - x0) + points[0]

    points = np.array([points[0], linear(x0 + dx), *
                      points[1:-1], linear(xn - dx), points[-1]])
    condlist = [x < x0, np.logical_and(x >= x0, x < xn), x >= xn]
    funclist = [linear, lambda z: bezier((z-x0)/dx2, points), linear]
    return np.piecewise(x, condlist, funclist)


def fourier(x, *params):
    params = np.array(params)
    assert len(params.shape) == 1
    n = len(params) - 3
    assert n % 2 == 0
    assert n >= 0
    T = params[0]
    m = params[1]
    offset = params[2]
    coeffs = params[3:]
    k = 2*np.pi / T
    ts = k * np.atleast_2d(np.arange(1, n//2+1)
                           ).transpose().dot(np.atleast_2d(x))
    s = np.sin(ts)
    c = np.cos(ts)
    return coeffs.dot(np.concatenate((s, c))) + m*x + offset


def load_data(file):
    rawdata = np.genfromtxt(file, delimiter=',', skip_header=1)
    t = rawdata[:, 0]
    VC = np.power(10, -rawdata[:, 1]/2.5)
    # 10^(-x/2.5)' = e^(ln10 * -x/2.5)' = -ln(10)/2.5 * e^(ln10 * -x/2.5)
    VCerr = np.log(10)/2.5 * np.power(10, -rawdata[:, 1]/2.5) * rawdata[:, 2]
    data = np.stack((t, VC, VCerr), axis=-1)
    return data


def plot_data(data, date, fit_funcs, fit_params, labels, star, filename=None):
    t, VC, VCerr = data.transpose()
    VCmax, VCmin, VCmean = VC.max(), VC.min(), VC.mean()
    VCerrmax = np.max(VCerr)

    #plt.errorbar(x=jul_to_greg(t), y=VC, yerr= VCerr, fmt='x')
    plt.figure(figsize=(14, 10), dpi=300)

    plt.errorbar(x=t, y=VC, yerr=VCerr, fmt='.',
                 elinewidth=.8, capsize=2, label='Daten')

    for i, fit_func in enumerate(fit_funcs):
        fit = fit_func(t, *(fit_params[i]))
        plt.plot(t, fit, label=labels[i])

    plt.grid(True, color='gray', linewidth=0.1)
    #plt.ylim(VCmin - 0.01 * VCmean - VCerrmax, VCmax + 0.01 * VCmean + VCerrmax)
    plt.legend()
    plt.title('Lichtkurve ' + star)
    plt.xlabel('Zeit in Tagen am ' + str(jul_to_greg(date).date()))
    plt.ylabel('Strahlungsflussverhältnis V-C')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    exodata = load_data('./Tres-1b/Tres-1b_V-C.csv')
    jdt_exo = int(exodata[:, 0].mean())
    exodata[:, 0] -= jdt_exo
    t_exo = exodata[:, 0]
    VC_exo = exodata[:, 1]
    vardata = load_data('./UCAC4_558-007313/UCAC4_558-007313_Light_Curve.csv')
    jdt_var = int(vardata[:, 0].mean())
    vardata[:, 0] -= jdt_var
    t_var = vardata[:, 0]
    VC_var = vardata[:, 1]

    p0tr = np.array([0.33, 1.0, 0.35, 1.0, 0.425, 1.0, 0.44, 1.0])
    pt0 = np.array([0., 1.0, 0.375, 0.2, 0.1, 0.05])
    pb0 = np.array([0.385, 0.425, 1.0, 0.9, 0.9, 0.9, 1.0])

    # Hier bitte symmetrischen Trapez-Fit
    paramstr, paramstrcov = opt.curve_fit(trapez, t_exo, VC_exo, p0=p0tr)
    paramst, paramstcov = opt.curve_fit(transit, t_exo, VC_exo, p0=pt0)
    paramsb, paramsbcov = opt.curve_fit(transit_bezier, t_exo, VC_exo, p0=pb0)
    plot_data(exodata, jdt_exo, [trapez, transit, transit_bezier], [paramstr, paramst, paramsb], [
              'Trapez-Fit', 'symmetrischer Trapez-Fit', 'Bézier-Fit'], 'Tres-1b', filename='Tres-1b.png')

    # Ordnung 7 sieht gut aus mit Gerade
    params6, params6cov = opt.curve_fit(fourier, t_var, VC_var, p0=[
                                        0.15, -0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    params2, params2cov = opt.curve_fit(fourier, t_var, VC_var, p0=[
                                        0.15, -0.1, 0.5, 0.5, 0.5, 0.5, 0.5])
    params1, params1cov = opt.curve_fit(fourier, t_var, VC_var, p0=[
                                        0.15, -0.1, 0.5, 0.5, 0.5])
    plot_data(vardata, jdt_var, [fourier, fourier, fourier], [params6, params2, params1], [
              'Fourier-Fit 6.Ordnung', 'Fourier-Fit 2.Ordnung', 'Fourier-Fit 1.Ordnung'], 'UCAC4 558-007131', filename='UCAC4.png')

    exoparams = paramst
    exostd = np.sqrt(np.diag(paramstcov))
    varparams = params6
    varstd = np.sqrt(np.diag(params6cov))

    print("Exoplanet-Transit:")
    print("Transittiefe:[relativer Strahlungsfluss]",
          exoparams[3], "+-", exostd[3])
    print("relative Transittiefe:", exoparams[3]/exoparams[1], "+-", np.sqrt(np.square(
        exostd[3]/exoparams[1]) + np.square(exoparams[3]*exostd[1]/exoparams[1]**2)))
    print("Transit-Duration[d]:", exoparams[4], "+-", exostd[4])
    print("Totalitätsdauer[d]:", exoparams[5], "+-", exostd[5])

    print("Bedeckungsveränderlicher:")
    print("Periodendauer:", varparams[0], "+-", varstd[0])
