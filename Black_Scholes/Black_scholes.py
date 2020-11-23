####################### Packages #######################
import numpy as np
from scipy.special import comb
import math
import matplotlib.pyplot as plt

####################### define u #######################
def uv(Rv, sig, hv):
    return Rv * hv + sig * np.sqrt(hv)

####################### define l #######################
def lv(Rv, sig, hv):
    return Rv * hv - sig * np.sqrt(hv)

####################### Function f #######################
def f(xval, Kv):
    val = Kv - xval
    return max(val, 0)

####################### Phi(x) #######################
def Phi(x):
    p = 0.231641900
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    t = 1 / (1 + p * x)
    val = np.exp((-1) * x**2 / 2)
    val = val * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
    pos = np.sqrt(2 * np.pi)
    return 1 - val / pos

####################### Theoritical #######################
def Put_theo(rv, Tv, Kv, Sv, sig):
    dp = np.log(Sv / Kv) + (rv + sig**2/2) * Tv
    dp = dp / (sig * np.sqrt(Tv))
    dm = np.log(Sv / Kv) + (rv - sig**2/2) * Tv
    dm = dm / (sig * np.sqrt(Tv))
    fst = np.exp((-1) * rv * Tv) * Kv * (1 - Phi(dm))
    snd = Sv * (1 - Phi(dp))
    return fst - snd

####################### x value at time t #######################
# input:
    # Uv: the u value
    # Lv: the l value
    # x0: the initial value
    # n: the time step
# output:
    # rlst: x value list at time ct
def xv(Uv, Lv, x0, n):
    rlst = []       # return list
    for i in range(n + 1):
        xval = x0 * (1 + Uv)**i * (1 + Lv)**(n - i)
        rlst.append(xval)
    return rlst

####################### Param #######################
S01 = 100
K = 100
r = 0.01
sigma = 0.1
T = 3
Nlst = list(range(T, 11))
xFlst = []
thlst = []
Nlst = list(range(T, 101))
for N in Nlst:
    h = T / N
    ####################### T_h #######################
    timelist = np.linspace(0, T, N + 1)

    ####################### u, l, and q #######################
    u = uv(r, sigma, h)
    l = lv(r, sigma, h)
    q = 0.5

    ####################### x value #######################
    xvlst = [[S01]]
    for i in range(len(timelist)):
        if i != 0:
            xvt = xv(u, l, S01, i)
            xvlst.append(xvt)

    ####################### backward induction #######################
    rtlst = list(reversed(timelist))
    Vlst = []
    for i in range(len(rtlst)):
        ilst = []
        if i == 0:
            for xval in xvlst[N - i]:
                ilst.append(f(xval, K))
        else:
            vilst = Vlst[-1]
            jlst = list(range(len(xvlst[N - i])))
            for j in jlst:
                v1 = q * vilst[j + 1]
                v2 = (1 - q) * vilst[j]
                val = (v1 + v2) / (1 + r * h)
                ilst.append(val)
        Vlst.append(ilst)
    xF = Vlst[-1][0]
    xFlst.append(xF)
    thlst.append(Put_theo(r, T, K, S01, sigma))

####################### Output Figure #######################
png_name = "./Output/put_T_{0}_maxN_{1}_sig_{2}_S01_{3}_r_{4}_K_{5}.png"
eps_name = "./Output/put_T_{0}_maxN_{1}_sig_{2}_S01_{3}_r_{4}_K_{5}.eps"
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('N', fontsize = 15)
ax.set_ylabel('Price', fontsize = 15)
ax.set_title("Convergence to the Black-Scholes put option price.", fontsize = 18)
ax.plot(Nlst, xFlst, label = "CRR", c = "m")
ax.scatter(Nlst, thlst, label = "BSM", c = "r", s = 1)
ax.grid(True)
ax.legend(loc = 'best')
plt.savefig(png_name.format(str(T), str(max(Nlst)), str(sigma), str(S01), str(r), str(K)), format = 'PNG')
plt.savefig(eps_name.format(str(T), str(max(Nlst)), str(sigma), str(S01), str(r), str(K)), format = 'EPS')