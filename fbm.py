import time
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
    FBM path generator given time points
    :param lambd: float
    :param alpha: float
    :param n: int
    :param x: 1D array representing time
    :param path_num: int
    :return: 2D array with shape (len(x), path_num)
    """
    x = np.tile(np.ravel(x), (path_num, 1)).T
    y = np.zeros(x.shape)

    for i in range(n+1):
        c = np.random.normal(0, 1, path_num).reshape((1, -1))
        a = np.random.uniform(0, 2*np.pi, path_num).reshape((1, -1))
        y += c * lambd ** (-i * alpha) * np.sin(lambd ** i * x + a)

    return y


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


# # 单个路径模拟
#
# lambd = 1.9
# alpha = 0.8  # 该值太小时会导致无法计算implied vol
# n = 30
# x = np.arange(0, 100, 0.01)
# path_num = 1
#
# y = fbm(lambd, alpha, n, x, path_num)
# plt.plot(x, y)
# plt.show()


# 多路径模拟与隐含波动率

def simulate():

    # 参数 begin
    s_start = 1  # 起始期权价格
    expiry = 0.5  # 到期时间（以年计）
    risk_free = 0  # 年化无风险利率，由于 fbm 中没有类似参数，这里设为 0
    strikes = np.arange(0.9, 2.5, 0.02)  # 行权价。strike在 0.9 以下时会无法求解波动率，待研究

    lambdas = np.arange(1.1, 4, 0.5)
    alphas = np.arange(0.3, 0.9, 0.1)  # 当 alpha 小于 0.3 时波动率会非常奇怪，待研究
    
    n = 30  # 级数的阶

    step = 0.002  # 两个相邻时间点的间隔
    x = np.arange(0, expiry+step, step)  # 在期权存续期上取的离散时间点
    path_num = 1000  # 模拟路径个数
    # 参数 end

    # 计算不同 lambda 的结果，绘制在不同的图上
    nrows = 3
    ncols = int(np.ceil(len(lambdas) / nrows))
    figsize = (12, 12)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = trim_axs(axes, len(lambdas))

    for ax, lambd in zip(axes, lambdas):

        # 在同一张图里计算不同 alpha 的结果，绘制成不同曲线
        for alpha in alphas:

            paths = fbm(lambd, alpha, n, x, path_num)  # 得到所有路径
            ends = paths[-1]  # 取所有路径的终点
            ends_exp = np.exp(ends)
            ends_exp_adj = ends_exp / np.average(ends_exp)  # normalization，保证股价是 martingale
            s_ends = s_start * ends_exp_adj

            # 计算不同 strike 下的 implied vol
            implied_vols = []

            for strike in strikes:

                payoffs = np.maximum(s_ends - strike, 0)
                payoff_pv = np.average(payoffs) * np.exp(-risk_free * expiry)

                implied_vol = vanilla_call_implied_vol(s_start, strike, expiry, risk_free, payoff_pv)
                implied_vols.append(implied_vol)

            ax.plot(strikes, implied_vols, label="alpha = " + str(round(alpha, 2)))

        ax.legend(loc="upper right")
        ax.set_title("lambda = " + str(round(lambd, 2)))

    plt.show()
    plt.savefig("./fig.png")


if __name__ == "__main__":

    time1 = time.time()

    simulate()

    time2 = time.time()
    print("Time used:", round(time2 - time1, 4), "seconds")
