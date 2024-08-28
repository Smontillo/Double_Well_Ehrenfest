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
# def DW_pop(ρ):
#     ρDW = np.zeros((par.nData, par.nDW))
#     ρ_tensor= ρ.reshape([par.nData, par.nc, par.nDW, par.nc, par.nDW])
#     ρDWf = np.trace(ρ_tensor, axis1 = 1, axis2 = 3)
#     for h in range(par.nData):
#         ρDW[h, :] = np.diag(ρDWf[h,:])
#     return ρDW
    
def DW_pop(ρ):
    ρDW = np.zeros((par.nData, par.nDW))
    for k in range(par.nData):
        ρ_tensor= ρ[k,:].reshape([par.nc, par.nDW, par.nc, par.nDW])
        ρDWf = np.trace(ρ_tensor, axis1 = 0, axis2 = 2)
        ρDW[k, :] = np.diag(ρDWf[:,:])
    return ρDW
# def cav_pop(ρ):
#     ρc = np.zeros((par.nData, par.nc))
#     ρ_tensor= ρ.reshape([par.nData, par.nc, par.nDW, par.nc, par.nDW])
#     ρcf = np.trace(ρ_tensor, axis1 = 2, axis2 = 4)
#     for h in range(par.nData):
#         ρc[h, :] = np.diag(ρcf[h,:])
#     return ρc

def cav_pop(ρ):
    ρc = np.zeros((par.nData, par.nc))
    for k in range(par.nData):
        ρ_tensor= ρ[k,:].reshape([par.nc, par.nDW, par.nc, par.nDW])
        ρDWf = np.trace(ρ_tensor, axis1 = 1, axis2 = 3)
        ρc[k, :] = np.diag(ρDWf[:,:])
    return ρc
# LOADING OF DATA
cpus = par.Cpus
Nst = par.nData
time = par.Sim_time[::par.nskip]
print(Nst, 'Number of steps')
ρ = np.zeros((Nst, par.nt**2), dtype=np.complex128)

for k in range(cpus):
    ρ += np.loadtxt(f'./data/rho_{k}.txt')
ρ /= cpus
ρ = np.real(ρ)

print(ρ[:10,9])
print(ρ[:10,45])
ρDW = DW_pop(ρ)
ρcav = cav_pop(ρ)

int_PL = np.zeros(par.nData-1)
int_PR = np.zeros(par.nData-1)
dt = time[1] - time[2]
for k in range(par.nData - 1):
    int_PL[k] = np.trapz(ρDW[:k,0], time[:k], dt)
    int_PR[k] = np.trapz(ρDW[:k,1], time[:k], dt)

X = (int_PL, int_PR)
popt, pcov = curve_fit(rates, X, ρDW[:-1,1], bounds = (0, [1E-5, 1E-5]))
print('Forward rate ->', popt[0], 'Backwards rate ->', popt[1])
print(popt[0] * par.fstoau)

# ===================================
color = ['#2980b9', '#c0392b' ,'#27ae60', '#8e44ad', '#e67e22', '#34495e']
# ========================================
fig, ax = plt.subplots(figsize = (4.5,4.5))
tot = np.sum(ρDW, axis = 1)
print(tot[-1])
label = ['|ν_L⟩', '|ν_R⟩', "|ν'_L⟩", "|ν'_R⟩", "Tot. Pop"]
for k in range(par.nDW):
    ax.plot(time/par.fstoau/1000, ρDW[:,k],  ls = '-', lw = 3, color = color[k],  label = f"{label[k]}", alpha = 0.8) 
ax.plot(time/par.fstoau/1000, tot, ls = '--', lw = 2)

np.savetxt('Lind_DW.txt', np.c_[time/par.fstoau/1000, ρDW])
# state_0 = np.loadtxt('./Deping_data/state_0.txt')
# state_1 = np.loadtxt('./Deping_data/state_1.txt')
# state_2 = np.loadtxt('./Deping_data/state_2.txt')
# state_3 = np.loadtxt('./Deping_data/state_3.txt')

# ax.plot(state_0[:,0], state_0[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
# ax.plot(state_1[:,0], state_1[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
# ax.plot(state_2[:,0], state_2[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
# ax.plot(state_3[:,0], state_3[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)

state_0 = np.loadtxt('../Double_Well/Deping_data/state_0.txt')
state_1 = np.loadtxt('../Double_Well/Deping_data/state_1.txt')
state_2 = np.loadtxt('../Double_Well/Deping_data/state_2.txt')
state_3 = np.loadtxt('../Double_Well/Deping_data/state_3.txt')

ax.plot(state_0[:,0], state_0[:,1], ls = '-.', lw = 1, color = 'red', alpha = 0.9)
ax.plot(state_1[:,0], state_1[:,1], ls = '-.', lw = 1, color = 'red', alpha = 0.9)
ax.plot(state_2[:,0], state_2[:,1], ls = '-.', lw = 1, color = 'red', alpha = 0.9)
ax.plot(state_3[:,0], state_3[:,1], ls = '-.', lw = 1, color = 'red', alpha = 0.9)

ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
ax.tick_params(which='minor', length=5, direction = 'in')
ax.set_xlabel('Time (ps)', fontsize = 20)
ax.set_ylabel('Population', fontsize = 20)


ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)

plt.savefig('images/pop_DW.png', dpi = 300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize = (4.5,4.5))
tot = np.sum(ρcav, axis = 1)
print(tot[-1])
label = np.arange(0,par.nc,1)
for k in range(par.nc):
    ax.plot(time/par.fstoau/1000, ρcav[:,k],  ls = '-', lw = 3, color = color[k],  label = f"{label[k]}", alpha = 0.8) 
ax.plot(time/par.fstoau/1000, tot, ls = '--', lw = 2)

ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
ax.tick_params(which='minor', length=5, direction = 'in')
ax.set_xlabel('Time (ps)', fontsize = 20)
ax.set_ylabel('Population', fontsize = 20)


ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)

plt.savefig('images/pop_cav.png', dpi = 300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize = (4.5,4.5))
tot = np.sum(ρDW, axis = 1)
label = ['|ν_L⟩, 0', '|ν_R⟩, 0', "|ν'_L⟩, 0", "|ν'_R⟩, 0"]
for k in range(par.nDW):
    ax.plot(time/par.fstoau/1000, ρ[:,(par.nDW * par.nc + 1) * k ],  ls = '-', color = color[k], lw = 3,  label = f"{label[k]}", alpha = 0.8) 

state_0 = np.loadtxt('../Double_Well/Deping_data/state_0.txt')
state_1 = np.loadtxt('../Double_Well/Deping_data/state_1.txt')
state_2 = np.loadtxt('../Double_Well/Deping_data/state_2.txt')
state_3 = np.loadtxt('../Double_Well/Deping_data/state_3.txt')

ax.plot(state_0[:,0], state_0[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
ax.plot(state_1[:,0], state_1[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
ax.plot(state_2[:,0], state_2[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
ax.plot(state_3[:,0], state_3[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9, label = 'Out. Cav.')

ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
ax.set_ylim(-0.01,0.12)
# ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
ax.tick_params(which='minor', length=5, direction = 'in')
ax.set_xlabel('Time (ps)', fontsize = 20)
ax.set_ylabel('Population', fontsize = 20)


ax.legend(title ='States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)

plt.savefig('images/states_0.png', dpi = 300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize = (4.5,4.5))
tot = np.sum(ρDW, axis = 1)
label = ['|ν_L⟩, 1', '|ν_R⟩, 1', "|ν'_L⟩, 1", "|ν'_R⟩, 1", "Tot. Pop"]
for k in range(par.nDW):
    ax.plot(time/par.fstoau/1000, ρ[:,(par.nDW * par.nc + 1) * (k + 4) ],  ls = '-', lw = 3, color = color[k],  label = f"{label[k]}", alpha = 0.8) 


ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
ax.tick_params(which='minor', length=5, direction = 'in')
ax.set_xlabel('Time (ps)', fontsize = 20)
ax.set_ylabel('Population', fontsize = 20)


ax.legend(title ='States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)

plt.savefig('images/states_1.png', dpi = 300, bbox_inches='tight')
plt.close()