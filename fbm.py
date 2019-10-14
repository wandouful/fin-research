import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def vanilla_call(s, k, t, r, v):
    if t > 0:
        d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / (v * np.sqrt(t))
        d2 = d1 - v * np.sqrt(t)
        nd1 = norm.cdf(d1)
        nd2 = norm.pdf(d2)
    else:
        nd1 = nd2 = 1. * (s > k)
    return nd1 * s - nd2 * np.exp(- t * r) * k


def vanilla_call_implied_vol(s, k, t, r, p):
    return fsolve(lambda v: vanilla_call(s, k, t, r, v) - p, np.ones(p.shape))


def fbm(lambd, alpha, n, x, path_num):
    """
    :param lambd: float
    :param alpha: float
    :param n: int
    :param x: 1D array representing time
    :param path_num: int
    :return: 2D array
    """
    x = np.tile(np.ravel(x), (path_num, 1)).T
    y = np.zeros(x.shape)

    for i in range(n+1):
        c = np.random.normal(0, 1, path_num).reshape((1, -1))
        a = np.random.uniform(0, 2*np.pi, path_num).reshape((1, -1))
        y += c * lambd ** (-i * alpha) * np.sin(lambd ** i * x + a)

    return y


# # 单个路径模拟
#
# lambd0 = 1.9
# alpha0 = 0.8  # 该值太小时会导致无法计算implied vol
# n0 = 30
# x0 = np.arange(0, 100, 0.01)
# path_num0 = 1
#
# y0 = fbm(lambd0, alpha0, n0, x0, path_num0)
# plt.plot(x0, y0)
# plt.show()


# 多路径模拟与隐含波动率

def simulate():

    # 参数 begin
    s_start = 1
    expiry = 0.5  # in year
    risk_free = 0.05  # annual

    lambd0 = 1.9
    alpha0 = 0.8
    n0 = 30
    x0 = np.arange(0, expiry, 0.001)
    path_num0 = 1000

    strike_prices = np.arange(0.9, 2.5, 0.005)
    # 参数 end

    # 得到所有路径
    paths = fbm(lambd0, alpha0, n0, x0, path_num0)
    ends = paths[-1]
    s_ends = s_start * np.exp(ends)

    # 计算不同 strike 下的 implied vol
    implied_vols = []

    for strike in strike_prices:

        payoffs = np.maximum(s_ends - strike, 0)
        payoff_pv = np.average(payoffs) * np.exp(-risk_free * expiry)

        implied_vol = vanilla_call_implied_vol(s_start, strike, expiry, risk_free, payoff_pv)
        implied_vols.append(implied_vol)

    implied_vols = np.array(implied_vols)
    
    return strike_prices, implied_vols, s_ends
    

strikes, vols, s = simulate()
    
plt.plot(strikes, vols)
plt.show()
