from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

fstoau = 41.341 
def rates(X, kf, kb):
    x1, x2 = X
    return kf * x1 - kb * x2 

state_0 = np.loadtxt('./state_0.txt')
state_1 = np.loadtxt('./state_1.txt')

st0 = interp1d(state_0[:,0], state_0[:,1])
st1 = interp1d(state_1[:,0], state_1[:,1])
time = np.arange(0,7,6E-3)
int_PL = np.zeros(len(time)-1)
int_PR = np.zeros(len(time)-1)

dt = time[1] - time[2]
for k in range(len(time) - 1):
    int_PL[k] = np.trapz(st0(time[:k]), time[:k], dt)
    int_PR[k] = np.trapz(st1(time[:k]), time[:k], dt)

X = (int_PL, int_PR)
popt, pcov = curve_fit(rates, X, st1(time[:-1]), bounds = (0, [1E-5, 1E-5]))
print('Forward rate ->', popt[0], 'Backwards rate ->', popt[1])
print(popt[0] * fstoau)