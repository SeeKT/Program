# Written by K.Toda, 2021/01/30
# Python 3.7.5, pip 20.3.3, pipenv version 2020.11.15

import numpy as np
from math import e
import scipy
from scipy import linalg
from scipy.spatial import distance
import matplotlib.pyplot as plt
from autograd import grad as nabla
from autograd import hessian
import os

base_dir = "./"
fig_dir = base_dir + "out_fig/"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
png_name = "{0}_{1}_{2}_{3}.png"
png_name = fig_dir + png_name
eps_name = "{0}_{1}_{2}_{3}.eps"
eps_name = fig_dir + eps_name


class calculate_iteration():

    def __init__(self, funcf, Xinit, eps, eta, p, smallvalue, rho, beta, gamma, Maxitr, ky):
        self.funcf = funcf
        self.Xinit = Xinit
        self.eps = eps
        self.eta = eta
        self.p = p
        self.smallvalue = smallvalue
        self.rho = rho
        self.beta = beta
        self.gamma = gamma
        self.Maxitr = Maxitr
        self.ky = ky

    # make_list
    def make_list(self, n):
        return [[] for i in range(n)]
    
    # 関数f1, グラフ描画用 (1)
    def f1fig(self, X1, X2):
        return X1**2 - 2*X1*X2 + X2**4/4 - X2**3/3
    
    # 関数f2, グラフ描画用 (2)
    def f2fig(self, X1, X2):
        return X1**3 + X2**3 - 3*X1*X2
    
    # 関数f3
    def f3fig(self, X1, X2):
        return -4*np.exp(-X1**2 - X2**2) - 2*np.exp(-(X1 + 4)**2 - (X2 + 4)**2)

    # 最急降下法(1step) (1-1)
    def steep_dec(self, curX):
        # 現在の勾配を求める
        grd = nabla(self.funcf)(curX).reshape(curX.shape)
        # 更新則 (epsilon固定)
        new_X = curX - self.eps * grd
        return new_X
    
    # 最急降下法(1step, 直線探索): (3)で実装
    def steep_dec_almijo(self, curX):
        # 現在の勾配を求める
        grd = nabla(self.funcf)(curX).reshape(curX.shape)
        # 直線探索 (Armijoのルール)
        l_k = 1
        while True:
            lft = self.funcf(curX + self.beta**l_k * (-1) * grd) - self.funcf(curX)
            rit = self.gamma * self.beta**l_k * grd.T @ ((-1) * grd)
            if lft > rit:
                l_k += 1
            else:
                eps = self.beta**l_k
                break
        # 更新則
        new_X = curX - eps * grd
        return new_X

    # モーメンタム法(1step) (1-2)
    def momentum(self, curX, prevX):
        # 現在の勾配を求める
        grd = nabla(self.funcf)(curX).reshape(curX.shape)
        # 更新則
        new_X = curX - self.eps * grd + self.p * (curX - prevX)
        return new_X

    # ニュートン法(1step) (1-3)
    def newton(self, curX):
        # 現在の勾配を求める
        grd = nabla(self.funcf)(curX).reshape(curX.shape)
        # 現在のHessianを求める
        hesse = hessian(self.funcf)(curX).reshape((curX.size, curX.size))
        new_X = curX - linalg.inv(hesse) @ grd
        return new_X

    # Adagrad: (3)
    def adagrad_iteration(self, curX, gradlist):
        # 現在の勾配を求める
        grd = nabla(self.funcf)(curX).reshape(curX.shape)
        gradlist.append(grd)
        upd_vec = []
        # 各成分ごとに
        for i in range(curX.size):
            # 分子: eta * (現在の勾配の第i成分)
            val = self.eta * grd[i][0]
            smm = 0
            # 分母: 過去の勾配の第i成分の二乗和の平方根
            for j in range(len(gradlist)):
                smm += gradlist[j][i][0]**2
            pos = np.sqrt(smm + self.smallvalue)     # 0で割らないようにsmallvalueを加える
            upd_vec.append(val / pos)
        upd_vec = np.array(upd_vec).reshape(curX.shape)
        # 更新則
        new_X = curX - upd_vec
        return new_X, gradlist

    # RMSprop: (3)
    def rmsprop_iteration(self, curX, vlist):
        # 現在の勾配を求める
        grd = nabla(self.funcf)(curX).reshape(curX.shape)
        # vの更新
        if len(vlist) == 0:
            curV = (1 - self.rho) * (grd * grd)              # grd * grd: Hadamard積
        else:
            prevV = vlist[-1]
            curV = self.rho * prevV + (1 - self.rho) * (grd * grd)
        vlist.append(curV)
        upd_vec = []
        # 各成分ごとに
        for i in range(curX.size):
            # 分子: epsilon * (現在の勾配の第i成分)
            val = self.eps * grd[i][0]
            smm = 0
            # 分母: (v + smallvalue)の平方根
            pos = np.sqrt(curV[i] + self.smallvalue)    # 0で割らないようにsmallvalueを加える
            upd_vec.append(val / pos)
        upd_vec = np.array(upd_vec).reshape(curX.shape)
        # 更新則
        new_X = curX - upd_vec
        return new_X, vlist

    # 更新
    def iteration_upd(self, keyword):
        lstX = [self.Xinit]; lstfX = [self.funcf(self.Xinit)]; gradlist = []; vlist = []
        itr = self.Maxitr; flag = 1
        for i in range(1, self.Maxitr):
            # 現在のx
            curX = lstX[-1]
            # 1ステップ前のx
            if len(lstX) == 1:
                prevX = lstX[-1]
            else:
                prevX = lstX[-2]
            # 1ステップ後のx
            if keyword == "steepest":
                newX = self.steep_dec(curX)
            elif keyword == "steepest_lin":
                newX = self.steep_dec_almijo(curX)
            elif keyword == "newton":
                newX = self.newton(curX)
            elif keyword == "momentum":
                newX = self.momentum(curX, prevX)
            elif keyword == "adagrad":
                newX, gradlist = self.adagrad_iteration(curX, gradlist)
            elif keyword == "rmsprop":
                newX, vlist = self.rmsprop_iteration(curX, vlist)
            else:
                print("Miss the keyword")
                return [], [], 0
            # 現在と1ステップ後の距離
            dst = distance.euclidean(newX, curX)
            if dst < 1e-6 and flag == 1:
                # 収束までの繰り返し数の表示
                print("Converse, roop: %d"%i)
                itr = i; flag = 0
            if dst < 1e+5:
                lstX.append(newX)
                lstfX.append(self.funcf(newX))
            # 発散したら表示する
            else:
                print("Diverse!")
                break
        return lstX, lstfX, itr
    
    # 関数の実行部
    def execute_iteration(self):
        ## Iterationの実行
        Xlst_s, flst_s, itr_s = self.iteration_upd(keyword = "steepest")
        Xlst_sl, flst_sl, itr_sl = self.iteration_upd(keyword = "steepest_lin")             # (3)
        Xlst_m, flst_m, itr_m = self.iteration_upd(keyword = "momentum")
        Xlst_n, flst_n, itr_n = self.iteration_upd(keyword = "newton")
        Xlst_a, flst_a, itr_a = self.iteration_upd(keyword = "adagrad")                     # (3)
        Xlst_r, flst_r, itr_r = self.iteration_upd(keyword = "rmsprop")                     # (3)
        print("Steepest: {}, Steepest(linear search): {}, Momentum: {}, Newton: {}, Adagrad: {}, RMSprop: {}".format(str(itr_s), str(itr_sl), str(itr_m), str(itr_n), str(itr_a), str(itr_r)))

        ## 関数の等高線の描画
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        if self.ky == 1:
            x1 = np.linspace(-2, 3, 51)
            x2 = np.linspace(-2, 3, 51)
            X1_grid, X2_grid = np.meshgrid(x1, x2)
            value = self.f1fig(X1_grid, X2_grid)
            cont = ax.contour(X1_grid, X2_grid, value, levels=[-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0])
        elif self.ky == 2:
            x1 = np.linspace(-0.5, 1.5, 51)
            x2 = np.linspace(-0.5, 1.5, 51)
            X1_grid, X2_grid = np.meshgrid(x1, x2)
            value = self.f2fig(X1_grid, X2_grid)
            cont = ax.contour(X1_grid, X2_grid, value, levels=[-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        elif self.ky == 3:
            x1 = np.linspace(-6, 2, 51)
            x2 = np.linspace(-6, 2, 51)
            X1_grid, X2_grid = np.meshgrid(x1, x2)
            value = self.f3fig(X1_grid, X2_grid)
            cont = ax.contour(X1_grid, X2_grid, value, levels=[-3.0, -2.0, -1.5, -1.0, -0.5, -0.1, -0.05, -0.01])
        cont.clabel(fmt='%1.2f')
        plt.savefig(png_name.format("contour_f", str(self.ky), str(self.Xinit[0][0]), str(self.Xinit[1][0])))
        plt.savefig(eps_name.format("contour_f", str(self.ky), str(self.Xinit[0][0]), str(self.Xinit[1][0])))

        ## get list of data
        Xval_s = self.make_list(self.Xinit.size)
        Xval_sl = self.make_list(self.Xinit.size)                       # (3)
        Xval_m = self.make_list(self.Xinit.size)
        Xval_n = self.make_list(self.Xinit.size)
        Xval_a = self.make_list(self.Xinit.size)                        # (3)
        Xval_r = self.make_list(self.Xinit.size)                        # (3)
        for i in range(len(Xlst_s)):
            for j in range(Xlst_s[i].size):
                Xval_s[j].append(Xlst_s[i][j])

        ################## (3) ##################
        for i in range(len(Xlst_sl)):
            for j in range(Xlst_sl[i].size):
                Xval_sl[j].append(Xlst_sl[i][j])
        #########################################

        for i in range(len(Xlst_m)):
            for j in range(Xlst_m[i].size):
                Xval_m[j].append(Xlst_m[i][j])
        for i in range(len(Xlst_n)):
            for j in range(Xlst_n[i].size):
                Xval_n[j].append(Xlst_n[i][j])
                
        ################## (3) ##################
        for i in range(len(Xlst_a)):
            for j in range(Xlst_a[i].size):
                Xval_a[j].append(Xlst_a[i][j])
        for i in range(len(Xlst_r)):
            for j in range(Xlst_r[i].size):
                Xval_r[j].append(Xlst_r[i][j])
        #########################################
        
        ## 各ステップのxの値の描画
        fig = plt.figure(figsize = (10, 5))
        plt.subplots_adjust(wspace=0.25, hspace=0.6)
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel("iteration step")
        ax1.set_ylabel(r"$x_1$")
        if self.ky == 1:
            ax1.set_ylim([-1, 3])
        elif self.ky == 2:
            ax1.set_ylim([-1, 1.5])
        ax1.plot(Xval_s[0], label = r"$x_1$ (Steepest)", c = "r")
        ax1.plot(Xval_sl[0], label = r"$x_1$ (Steepest, Armijo)", c = "k")              # (3)
        ax1.plot(Xval_m[0], label = r"$x_1$ (Momentum)", c = "g")
        ax1.plot(Xval_n[0], label = r"$x_1$ (Newton)", c = "y")
        ax1.plot(Xval_a[0], label = r"$x_1$ (Adagrad)", c = "b")                        # (3)
        ax1.plot(Xval_r[0], label = r"$x_1$ (RMSprop)", c = "m")                        # (3)
        ax1.legend(loc = 'best')
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel("iteration step")
        ax2.set_ylabel(r"$x_2$")
        if self.ky == 1:
            ax2.set_ylim([-1, 3])
        elif self.ky == 2:
            ax2.set_ylim([-1, 1.5])
        ax2.plot(Xval_s[1], label = r"$x_2$ (Steepest)", c = "r")
        ax2.plot(Xval_sl[1], label = r"$x_2$ (Steepest, Armijo)", c = "k")              # (3)
        ax2.plot(Xval_m[1], label = r"$x_2$ (Momentum)", c = "g")
        ax2.plot(Xval_n[1], label = r"$x_2$ (Newton)", c = "y")
        ax2.plot(Xval_a[1], label = r"$x_2$ (Adagrad)", c = "b")                        # (3)
        ax2.plot(Xval_r[1], label = r"$x_2$ (RMSprop)", c = "m")                        # (3)
        ax2.legend(loc = 'best')
        plt.savefig(png_name.format("x-itr_f", str(self.ky), str(self.Xinit[0][0]), str(self.Xinit[1][0])))
        plt.savefig(eps_name.format("x-itr_f", str(self.ky), str(self.Xinit[0][0]), str(self.Xinit[1][0])))


        ## 経路のプロット
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ### 関数の等高線の描画
        if self.ky == 1:
            x1 = np.linspace(-1, 3, 51)
            x2 = np.linspace(0, 3, 51)
            X1_grid, X2_grid = np.meshgrid(x1, x2)
            value = self.f1fig(X1_grid, X2_grid)
            cont = ax.contour(X1_grid, X2_grid, value, levels=[-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0])
        elif self.ky == 2:
            x1 = np.linspace(-0.5, 1.5, 51)
            x2 = np.linspace(-0.5, 1.5, 51)
            X1_grid, X2_grid = np.meshgrid(x1, x2)
            value = self.f2fig(X1_grid, X2_grid)
            cont = ax.contour(X1_grid, X2_grid, value, levels=[-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        elif self.ky == 3:
            x1 = np.linspace(-6, 2, 51)
            x2 = np.linspace(-6, 2, 51)
            X1_grid, X2_grid = np.meshgrid(x1, x2)
            value = self.f3fig(X1_grid, X2_grid)
            cont = ax.contour(X1_grid, X2_grid, value, levels=[-3.0, -2.0, -1.5, -1.0, -0.5, -0.1, -0.05, -0.01])
        cont.clabel(fmt='%1.2f')

        ## 点の描画 (plot version)
        ax.plot(Xval_s[0], Xval_s[1], c = 'r', label = "Steepest")
        ax.plot(Xval_sl[0], Xval_sl[1], c = 'k', label = "Steepest (Armijo)")           # (3)
        ax.plot(Xval_m[0], Xval_m[1], c = 'g', label = "Momentum")
        ax.plot(Xval_n[0], Xval_n[1], c = 'y', label = "Newton")
        ax.plot(Xval_a[0], Xval_a[1], c = 'b', label = "Adagrad")                       # (3)
        ax.plot(Xval_r[0], Xval_r[1], c = 'm', label = "RMSprop")                       # (3)
        ax.legend(loc = 'best')
        plt.savefig(png_name.format("iteration_plot_f", str(self.ky), str(self.Xinit[0][0]), str(self.Xinit[1][0])))
        plt.savefig(eps_name.format("iteration_plot_f", str(self.ky), str(self.Xinit[0][0]), str(self.Xinit[1][0])))



################################# Main #################################
# (1): f1(x1, x2) = x1^2 - 2x1x2 + x2^4/4 - x2^3/3
f1 = lambda X: (X[0]**2 - 2*X[0]*X[1] + X[1]**4/4 - X[1]**3/3)[0]
# (2): 
    # f2(x1, x2) = x1^3 + x2^3 - 3x1x2
    # f3(x1, x2) = -4e^(-x1^2-x2^2) -2e^(-(x1+4)^2 - (x2+4)^2)
f2 = lambda X: (X[0]**3 + X[1]**3 - 3*X[0]*X[1])[0]
f3 = lambda X: (-4*e**(-X[0]**2 - X[1]**2) - 2*e**(-(X[0] + 4)**2 - (X[1] + 4)**2))[0]
X01 = np.array([
    [-0.5],
    [1.5]
])
X02 = np.array([
    [-0.25],
    [0.75]
])
X03 = np.array([
    [-2.0],
    [-2.0]
])
funclist = [f1, f2, f3]; X0list = [X01, X02, X03]
for i in range(len(funclist)):
    Optim = calculate_iteration(funcf=funclist[i], Xinit=X0list[i], eps=1e-2, eta=1e-1, p=0.9, smallvalue=1e-7, rho=0.9, beta=0.1, gamma=1e-5, Maxitr=500, ky=i+1)
    Optim.execute_iteration()