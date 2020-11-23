################################### Packages ###################################
import numpy as np
import matplotlib.pyplot as plt

################################### Parameter ###################################
a = 1.0                     # parameter a
K = 3000                    # the number of repetitions
delt = 0.01                 # step width
timelist = list(range(K))   # time list
timestep = np.arange(0.0, K * delt, delt)   # timestep list
x10 = 0.9                   # initial state
x20 = 0.5                   # initial state

################################### Hamiltonian ###################################
def dp(q, p):
    return p

def dq(q, p):
    return a * np.sin(q)

def dVq(q):
    return a * np.sin(q)

def dTp(p):
    return p

def Ham(q, p):
    return 0.5 * p**2 - a * np.cos(q) + a

################################### Dissipative ###################################
def f1(x1, x2):
    return x2

def f2(x1, x2):
    return (-1) * a * np.sin(x1) - b * x2

################################### Explicit Euler ###################################
def euler(q, p, dt):
    q_new = q + dt * dp(q, p)
    p_new = p - dt * dq(q, p)
    h = Ham(q_new, p_new)
    return q_new, p_new, h

################################### Runge-Kutta ###################################
def rk(x1, x2, h):
    k11 = dp(x1, x2)
    k12 = (-1) * dq(x1, x2)
    k21 = dp(x1 + (k11 * h / 2), x2 + (k12 * h / 2))
    k22 = (-1) * dq(x1 + (k11 * h / 2), x2 + (k12 * h / 2))
    k31 = dp(x1 + (k21 * h / 2), x2 + (k22 * h / 2))
    k32 = (-1) * dq(x1 + (k21 * h / 2), x2 + (k22 * h / 2))
    k41 = dp(x1 + (k31 * h), x2 + (k32 * h))
    k42 = (-1) * dq(x1 + (k31 * h), x2 + (k32 * h))
    x1_new = x1 + (h / 6) * (k11 + 2 * k21 + 2 * k31 + k41)
    x2_new = x2 + (h / 6) * (k12 + 2 * k22 + 2 * k32 + k42)
    ham = Ham(x1_new, x2_new)
    x1, x2 = x1_new, x2_new
    return x1, x2, ham
    
################################### Symplectic Euler ###################################
def symp(q, p, dt):
    p_new = p - dt * dVq(q)
    q_new = q + dt * dTp(p_new)
    h = Ham(q_new, p_new)
    return q_new, p_new, h

################################### Calculation ###################################
# explicit euler
x1_e = [x10]
x2_e = [x20]
h_e = [Ham(x10, x20)]
# runge-kutta
x1_r = [x10]
x2_r = [x20]
h_r = [Ham(x10, x20)]
# symplectic euler
x1_s = [x10]
x2_s = [x20]
h_s = [Ham(x10, x20)]

for k in timelist:
    if k != 0:
        # x value at time k
        x1ec = x1_e[-1]
        x2ec = x2_e[-1]
        x1rc = x1_r[-1]
        x2rc = x2_r[-1]
        x1sc = x1_s[-1]
        x2sc = x2_s[-1]
        # update x1 and x2
        x1en, x2en, he = euler(x1ec, x2ec, delt)
        x1rn, x2rn, hr = rk(x1rc, x2rc, delt)
        x1sn, x2sn, hs = symp(x1sc, x2sc, delt)
        # append to the list
        x1_e.append(x1en)
        x2_e.append(x2en)
        h_e.append(he)
        x1_r.append(x1rn)
        x2_r.append(x2rn)
        h_r.append(hr)
        x1_s.append(x1sn)
        x2_s.append(x2sn)
        h_s.append(hs)


################################### Output ###################################
png_name = "./Output/{0}_{1}_init_{2}_{3}.png"
eps_name = "./Output/{0}_{1}_init_{2}_{3}.eps"
################################### Explicit euler ###################################
# time
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(timestep, x1_e, c = 'purple', label = 'x1', s = 1)
ax.scatter(timestep, x2_e, c = 'blue', label = 'x2', s = 1)
ax.set_xlabel('time')
ax.set_ylabel('x1 and x2')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("euler", "time", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("euler", "time", str(x10), str(x20)), format = "EPS")

# trajectory
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x1_e, x2_e, c = 'red', label = 'Trajectory', s = 1)
ax.scatter(x10, x20, c = 'black', label = 'Initial State', s = 5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("euler", "trajectory", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("euler", "trajectory", str(x10), str(x20)), format = "EPS")

# Hamiltonian
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(timestep, h_e, c = 'red', label = 'Hamiltonian', s = 1)
ax.set_xlabel('time')
ax.set_ylabel('Hamiltonian')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("euler", "Ham", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("euler", "Ham", str(x10), str(x20)), format = "EPS")

################################### Runge-Kutta ###################################
# time
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(timestep, x1_r, c = 'purple', label = 'x1', s = 1)
ax.scatter(timestep, x2_r, c = 'blue', label = 'x2', s = 1)
ax.set_xlabel('time')
ax.set_ylabel('x1 and x2')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("rk", "time", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("rk", "time", str(x10), str(x20)), format = "EPS")

# trajectory
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x1_r, x2_r, c = 'red', label = 'Trajectory', s = 1)
ax.scatter(x10, x20, c = 'black', label = 'Initial State', s = 5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("rk", "trajectory", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("rk", "trajectory", str(x10), str(x20)), format = "EPS")

# Hamiltonian
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(timestep, h_r, c = 'red', label = 'Hamiltonian', s = 1)
ax.set_xlabel('time')
ax.set_ylabel('Hamiltonian')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("rk", "Ham", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("rk", "Ham", str(x10), str(x20)), format = "EPS")

################################### Symplectic euler ###################################
# time
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(timestep, x1_s, c = 'purple', label = 'x1', s = 1)
ax.scatter(timestep, x2_s, c = 'blue', label = 'x2', s = 1)
ax.set_xlabel('time')
ax.set_ylabel('x1 and x2')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("symp", "time", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("symp", "time", str(x10), str(x20)), format = "EPS")

# trajectory
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x1_s, x2_s, c = 'red', label = 'Trajectory', s = 1)
ax.scatter(x10, x20, c = 'black', label = 'Initial State', s = 5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("symp", "trajectory", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("symp", "trajectory", str(x10), str(x20)), format = "EPS")

# Hamiltonian
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(timestep, h_s, c = 'red', label = 'Hamiltonian', s = 1)
ax.set_xlabel('time')
ax.set_ylabel('Hamiltonian')
ax.legend(loc = 'best')
ax.grid(True)
plt.savefig(png_name.format("symp", "Ham", str(x10), str(x20)), format = "PNG")
plt.savefig(eps_name.format("symp", "Ham", str(x10), str(x20)), format = "EPS")