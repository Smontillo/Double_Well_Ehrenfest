import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time as tm
import parameters as par
from scipy.optimize import curve_fit
# ===================================
def rates(X, kf, kb):
    x1, x2 = X
    return kf * x1 - kb * x2 

# LOADING OF DATA
cpus = par.Cpus
Nst = par.nData
time = par.Sim_time[::par.nskip]
print(Nst, 'Number of steps')
ρ = np.zeros((Nst, par.nDW), dtype=np.complex128)
test = np.zeros((Nst,2), dtype=np.complex128)
for k in range(cpus):
    ρ += np.loadtxt(f'./data/rho_{k}.txt')
ρ /= cpus
ρ = np.real(ρ)

# =============================================
int_PL = np.zeros(par.nData-1)
int_PR = np.zeros(par.nData-1)
dt = time[1] - time[2]
for k in range(par.nData - 1):
    int_PL[k] = np.trapz(ρ[:k,0], time[:k], dt)
    int_PR[k] = np.trapz(ρ[:k,1], time[:k], dt)

X = (int_PL, int_PR)
popt, pcov = curve_fit(rates, X, ρ[:-1,1], bounds = (0, [1E-5, 1E-5]))
print('Forward rate ->', popt[0], 'Backwards rate ->', popt[1])
print(popt[0] * par.fstoau)

plt.plot(int_PL)
plt.plot(int_PR)
plt.savefig('images/test_rates.png')
plt.close()
# ===================================
color = ['#3498db', '#e74c3c' ,'#1abc9c', '#9b59b6', '#e67e22', '#34495e']
# ========================================
fig, ax = plt.subplots(figsize = (4.5,4.5))

tot = np.sum(ρ, axis = 1)
ax.plot(time/par.fstoau/1000, ρ[:,0],  ls = '-', lw = 3, color = color[0],   label = r'$|\nu_L⟩$', alpha = 0.8) 
ax.plot(time/par.fstoau/1000, ρ[:,1],  ls = '-', lw = 3, color = color[1],   label = r'$|\nu_R⟩$', alpha = 0.8) 
ax.plot(time/par.fstoau/1000, ρ[:,2],  ls = '-', lw = 3, color = color[2],  label = r"$|\nu'_L⟩$", alpha = 0.8) 
ax.plot(time/par.fstoau/1000, ρ[:,3],  ls = '-', lw = 3, color = color[3],  label = r"$|\nu'_R⟩$", alpha = 0.8) 
ax.plot(time/par.fstoau/1000,tot,      ls = '--',lw = 2, color = color[-1], label = r"$Tot. Pop$", alpha = 0.8) 
# ax.plot(time[:-1]/par.fstoau/1000, rates(X, popt[0], popt[1]), ls = '-', color = '#f1c40f')

np.savetxt('outside_data.txt', np.c_[time/par.fstoau/1000, ρ])

state_0 = np.loadtxt('./Deping_data/state_0.txt')
state_1 = np.loadtxt('./Deping_data/state_1.txt')
state_2 = np.loadtxt('./Deping_data/state_2.txt')
state_3 = np.loadtxt('./Deping_data/state_3.txt')

ax.plot(state_0[:,0], state_0[:,1], ls = '-', lw = 2, color = 'black', alpha = 0.9, label = "Deping's Data")
ax.plot(state_1[:,0], state_1[:,1], ls = '-', lw = 2, color = 'black', alpha = 0.9)
ax.plot(state_2[:,0], state_2[:,1], ls = '-', lw = 2, color = 'black', alpha = 0.9)
ax.plot(state_3[:,0], state_3[:,1], ls = '-', lw = 2, color = 'black', alpha = 0.9)

ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
ax.tick_params(which='minor', length=5, direction = 'in')
ax.set_xlabel('Time (fs)', fontsize = 20)
ax.set_ylabel('Population', fontsize = 20)


ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)

plt.savefig('images/pop.png', dpi = 300, bbox_inches='tight')
plt.close()