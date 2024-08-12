import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time as tm
import parameters as par
# ===================================
# LOADING OF DATA
cpus = 3 #par.Cpus
Nst = par.NSteps
print(Nst, 'Number of steps')
ρ = np.zeros((Nst, par.nDW**2), dtype=np.complex128)
for k in range(cpus):
    ρ += np.loadtxt(f'./data/rho_{k}.txt', dtype = 'complex')
ρ /= cpus


# ===================================
color = ['#3498db', '#e74c3c' ,'#1abc9c', '#9b59b6', '#e67e22', '#34495e']
# ========================================
fig, ax = plt.subplots(figsize = (4.5,4.5))

ax.plot(par.Sim_time/par.fstoau/1000, ρ[:,0],  ls = '-', lw = 3, color = color[0], label = r'$|\nu_L⟩$', alpha = 0.8) #r'$\rho_{\nu 6}$'
ax.plot(par.Sim_time/par.fstoau/1000, ρ[:,5],  ls = '-', lw = 3, color = color[1], label = r'$|\nu_R⟩$', alpha = 0.8) #r'$\rho_{\nu 6}$'
ax.plot(par.Sim_time/par.fstoau/1000,ρ[:,10],  ls = '-', lw = 3, color = color[2], label = r"$|\nu'_L⟩$", alpha = 0.8) #r'$\rho_{\nu 6}$'
ax.plot(par.Sim_time/par.fstoau/1000,ρ[:,15],  ls = '-', lw = 3, color = color[3], label = r"$|\nu'_R⟩$", alpha = 0.8) #r'$\rho_{\nu 6}$'

state_0 = np.loadtxt('./Deping_data/state_0.txt')
state_1 = np.loadtxt('./Deping_data/state_1.txt')
state_2 = np.loadtxt('./Deping_data/state_2.txt')
state_3 = np.loadtxt('./Deping_data/state_3.txt')

ax.plot(state_0[:,0], state_0[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
ax.plot(state_1[:,0], state_1[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
ax.plot(state_2[:,0], state_2[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
ax.plot(state_3[:,0], state_3[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)

ax.set_xlim(par.Sim_time[0]/par.fstoau/1000,par.Sim_time[-1]/par.fstoau/1000)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
ax.tick_params(which='minor', length=5, direction = 'in')
ax.set_xlabel('Time (fs)', fontsize = 20)
ax.set_ylabel('Population', fontsize = 20)


ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 11, handlelength=1, title_fontsize = 11, labelspacing = 0.2)

plt.savefig('images/pop.png', dpi = 300, bbox_inches='tight')
plt.close()
