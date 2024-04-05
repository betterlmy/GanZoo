import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy
import matplotlib.pyplot as plt


def generate_cosine_schedule(T, s=0.008):
    print("generate_cosine_schedule")

    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)
    t = [value for value in range(0, T + 1)]
    for i in t:
        alphas.append(f(i, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    print("generate_linear_schedule")
    betas = np.linspace(low * 1000 / T, high * 1000 / T, T)
    return betas


def generate_DNS_schedule(T, low, high, off=5):
    # Dynamic negative square.
    print("generate_DNS_schedule, T =", off)
    low = low * 1000 / T
    high = high * 1000 / T
    t = [value for value in range(0, T)]
    wt = []
    for i in t:
        w = -((i + off) / (T + 1 + off)) ** 2 + 1  # T+1防止最后失效
        wt.append(w)
    wt = np.array(wt)
    assert (high > low and low >= 0 and high <= 1), "high > low and low >= 0 and high <= 1"
    betas = (1 - wt) * (high - low) + low
    return betas


def get_alphas_cumprod(betas):
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)  # 累乘
    return alphas_cumprod


def test_schedule():
    """测试generate_schedule"""
    T = 2000
    betas_cos = generate_cosine_schedule(T)
    betas_lin = generate_linear_schedule(T, 0.0001, 0.02)
    betas_DNS5 = generate_DNS_schedule(T, 0.0001, 0.02, 5)
    betas_DNS500 = generate_DNS_schedule(T, 0.0001, 0.02, 500)
    betas_DNST = generate_DNS_schedule(T, 0.0001, 0.02, T)

    fig, ax = plt.subplots()
    ax.plot(np.arange(T), get_alphas_cumprod(betas_cos), label='Cosine', color='r')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_lin), label='Linear', color='g')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_DNS5), label='DNS5', color='c')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_DNS500), label='DNS500', color='y')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_DNST), label='DNST', color='k')

    plt.xlabel('TimeStep (t)')
    plt.ylabel('α_bar')
    # plt.title('Decay Schedule')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("alpha_bar_schedule.svg")


if __name__ == '__main__':
    test_schedule()
    pass
