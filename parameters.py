#!/software/anaconda3/2020.11/bin/python
#SBATCH -p debug
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=qbath.out
#SBATCH --error=qbath.err
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
# ==================================

# FUNCTIONS
# ==================================
# DOUBLE WELL R IN ENERGY BASIS
# @nb.jit(nopython=True, fastmath=True)
def Rx(nDW):
    pos_DW = np.zeros((nDW,nDW), dtype = np.complex128)
    for j in range(nDW):
        for i in range(nDW):
            avg_pos = diaV[:,j].conjugate() * x0 * diaV[:,i]
            pos_DW[j,i] = np.trapz(avg_pos,x0,dx)
    return pos_DW

# DOUBLE WELL POTENTIAL ENERGY
# @nb.jit(nopython=True, fastmath=True)
def DW(x,m,wDW):
    m = 1
    Eb = 2250 * cmtoau
    V = -(wDW**2 / 2) * x**2 + (wDW**4 / (16 * Eb)) * x**4
    return V - min(V)

# KINETIC ENERGY
# @nb.jit(nopython=True, fastmath=True)
def T(x,m):
    dx = x[0] - x[1]
    N = len(x)
    K = np.pi/dx
    Kin = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Kin[i,j] = K**2/3 * (1 + 2/N**2)
            else:
                Kin[i,j] = 2*K**2/N**2 * (-1)**(j-i)/(np.sin(np.pi * (j-i)/N))**2 
    return 1/(2*m) * Kin

# DISCRETE VARIABLE REPRESENTATION FUNCTION
# @nb.jit(nopython=True, fastmath=True)
def DVR(x,m,wDW):
    V = DW(x,m,wDW)
    V = np.diag(V)
    K = T(x,m)
    E, V = np.linalg.eigh(V+K)
    return E, V

# DRUDE - LORENTZ SPECTRAL DENSITY
# @nb.jit(nopython=True, fastmath=True)
def J_DrudeL(λ, γ, ω):
    return (2 * γ * λ * ω) / (ω**2 + γ**2)

# BATH PARAMETERS
# @nb.jit(nopython=True, fastmath=True)
def BathParam(λD, γD, N, num):    
    ωj = np.zeros((N))
    cj = np.zeros((N), dtype = np.complex128)

    if num == False:
    # ANALYTIC DISCRETIZATION OF THE DRUDE - LORENTZ SPECTRAL DENSITY 
    # Huo. P., et al (Mol. Phys. 2012, 110, 1035–1052)
        arr = np.arange(0,N,1) + 1
        ω_max = 10 * γD
        ωj[:] = γD * np.tan(arr/N * np.arctan(ω_max/γD))
        cj[:] = 2 * ωj[:] * np.sqrt(λD * np.arctan(ω_max/γD)/(np.pi * N))
    
    else:
        ω  = np.linspace(1E-10,100*γD,50000)                   # FREQUENCY SCAN FOR BATH FREQUENCIES
        dω = ω[1] - ω[0]
    # NUMERICAL DISCRETIZATION OF SPECTRAL DENSITY
    # Walters, P. L.; et al.  J Comput Chem 2017, 38 (2), 110–115. https://doi.org/10.1002/jcc.24527.

        J = J_DrudeL(λD, γD, ω)  
    
        Fω = np.zeros(len(ω))
        for i in range(len(ω)):
            Fω[i] = (4/np.pi) * np.sum(J[:i]/ω[:i]) * dω

        λs =  Fω[-1]
        for i in range(N):
            costfunc = np.abs(Fω-(((float(i)+0.5)/float(N))*λs))
            m = np.argmin((costfunc))
            ωj[i] = ω[m]
        cj[:] = ωj[:] * ((λs/(2*float(N)))**0.5)
    return cj, ωj

# BOSONIC CREATION OPERATOR
# @nb.jit(nopython=True, fastmath=True)
def creation(n):
    a = np.zeros((n,n), dtype = np.complex128)
    b = np.array([(x+1)**0.5 for x in range(n)], dtype = np.complex128)
    np.fill_diagonal(a[1:], b)
    return a

# ∂H/∂x_i - POSITION INDEPENDENT PART
# @nb.jit(nopython=True, fastmath=True)
def dHij_cons():
    dHij         = np.zeros((ndof, nDW, nDW), dtype = np.complex128)
    dHij[:,:,:] -= np.kron(cj, R).T.reshape(ndof, nDW, nDW)
    return dHij

# ELECTRONIC HAMILTONIAN 
# CONTRUCTED IN THE IN THE DIABATIC BASIS WITH 4 VIBRATIONAL STATES |ν_L⟩, |ν_R⟩, |ν'_L⟩, |ν'_R⟩ 
# @nb.jit(nopython=True, fastmath=True)
def Hel_cons():
    R2 = R @ R                                          # Rx^2
    H  = np.zeros((nDW, nDW), dtype = np.complex128)    
    H += np.diag(diaE)                                  # VIBRATIONAL STATES ENERGY | GROUND STATE ENERGY IS SUBSTRACTED
    H[0,1] += (EDW[1] - EDW[0])/2                       # |ν_L⟩ - |ν_R⟩ coupling    Δ  = (E[0] - E[1])/2
    H[1,0] += (EDW[1] - EDW[0])/2                       
    H[2,3] += (EDW[3] - EDW[2])/2                       # |ν'_L⟩ - |ν'_R⟩ coupling  Δ' = (E[2] - E[3])/2
    H[3,2] += (EDW[3] - EDW[2])/2                       
    H      += np.sum(cj**2/ωj**2) * R2/2
    return H 

'''
    SIMULATION PARAMETERS 
        Hu. D., et al. (J. Phys. Chem. Lett. 2023, 14 (49), 11208–11216. https://doi.org/10.1021/acs.jpclett.3c02985.)
'''

# PHYSICAL CONSTANTS
# ==================================
fstoau = 41.341                           # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK  = 3.1577464e+05 
temp   = 300 / autoK
β      = 1 / temp 

# SYSTEM PARAMETERS ==================================
M = 1.0                                                   # R0 MASS
nDW = 4                                                   # NUMBER OF VIBRATIONAL STATES IN DW
wDW = 1000 * cmtoau                                       # DW BARRIER FREQUENCY
N = 1024                                                  # NUMBER OF POINTS THAT DISCRETIZE R0 FOR DVR
L = 100.0                                                 # UPPER AND LOWER R0 LIMIT [-L, L]
x0 = np.linspace(-L,L,N)                                  # R0
dx = x0[0] - x0[1]                                        
EDW, VDW = DVR(x0,M,wDW)                                  # EIGENENERGIES AND EIGENSTATES FOR THE DW 
Normx = np.trapz(VDW[:,0].conjugate() * VDW[:,0],x0,dx)    
VDW = VDW/(Normx)**0.5                                    # NORMALIZE THE EIGENSTATES
VDW = -1.0 * np.array(VDW, dtype = np.complex128)         # EIGENSTATES ARE IN THE OPPOSITE DIRECTION

# DIABATIZATION OF  VIBRATIONAL STATES
# EIGENSTATES
diaV = np.zeros((len(VDW[:,0]), 4), dtype = np.complex128)
# |ν_L⟩ = (|0⟩ + |1⟩)/√2             |ν_R⟩ = (|0⟩ - |1⟩)/√2   
diaV[:,0], diaV[:,1] = (VDW[:,0] + VDW[:,1])/2**0.5, (VDW[:,0] - VDW[:,1])/2**0.5
# |ν'_L⟩ = (|2⟩ + |3⟩)/√2            |ν'_R⟩ = (|2⟩ - |3⟩)/√2   
diaV[:,2], diaV[:,3] = -(VDW[:,2] + VDW[:,3])/2**0.5, -(VDW[:,2] - VDW[:,3])/2**0.5
# EIGENENERGIES
diaE = np.zeros((4), dtype = np.complex128)
# E[ν_L] = E[ν_R] = (E[0] + E[1])/2 
diaE[0], diaE[1] = (EDW[0] + EDW[1])/2, (EDW[0] + EDW[1])/2
# E[ν'_L] = E[ν'_R] = (E[2] + E[3])/2 
diaE[2], diaE[3] = (EDW[2] + EDW[3])/2, (EDW[2] + EDW[3])/2
diaE -= diaE[0]
# POSITION OPERATOR
R = Rx(nDW)

# INITIAL STATE ==================================
# SYSTEM IS INITIALIZED IN THE REACTANT STATE |ν_L⟩
ρ0 = np.zeros((nDW,nDW), dtype = np.complex128)
ρ0[0,0] = 1.0 + 0 * 1j

# SIMULATION PARAMETERS ==============================
parallel = True
Cpus     = 50                                              # NUMBER THE CPUS USE FOR PARALLELIZATION
NTraj    = 500                                            # NUMBER OF TRAJECTORIES
tf       = 2000 * fstoau                                   # SIMULATION TIME IN FEMTOSECONDS
dtN      = 6                                               # NUCLEAR TIME STEP
NSteps   = int(tf/dtN)                                     # NUMBER OF SIMULATION STEPS
Sim_time = np.array([(x * dtN) for x in range(NSteps)])    # SIMULATION TIMES ARRAY
Estep    = 30                                              # NUMBER OF ELECTRONIC STEPS PER NUCLEAR TIME STEP ⇒ MUST BE EVEN!!!!
dtE      = dtN/Estep                                       # ELECTRONIC TIME STEP
nskip    = 5                                               # FRAME SAVED ⇒ HAS NOT BEEN PROGRAM YET, MUST BE 1!!!!

if NSteps%nskip == 0:
    nData = NSteps // nskip + 0
else :
    nData = NSteps // nskip + 1

# BATH PARAMETERS ==============================
ndof   = 300                                               # NUMBER OF BATH OSCILLATORS
γD     = 200 * cmtoau                                      # BATH CHARACTERISTIC FREQUENCY   
λD     = 0.1 * wDW * γD/2                                  # BATH REORGANIZATION ENERGY  
num    = False                                             # DISCRETIZATION OF THE SPECTRAL DENSITY | True ⇒ Numerical | False ⇒ Analytical
cj, ωj = BathParam(λD, γD, ndof, num)                      # BATH COUPLINGS AND FREQUENCIES
# TIME INDEPENDENT FUNCTIONS ==============================
Hel   = Hel_cons()                                         # ELECTRONIC HAMILTONIAN | INDEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
dHij = dHij_cons()                                         # ∂H/∂x_i                | INDEPENDENT OF THE POSITION OF THE BATH OSCILLATOR


if __name__ == '__main__': 
    print(np.real(np.round(Hel/cmtoau,3)))
    print('================')
    print(np.real(np.round(R,3)))
    print('================')
    print(np.real(np.sum((cj/ωj)**2)*np.dot(R,R)/2)/cmtoau)
    J = J_DrudeL(λD, γD, np.linspace(0,2500,200) * cmtoau)
    for n in range(ndof):
        plt.axvline(ωj[n]/cmtoau, ls = '-.', color = 'black', lw = 1)
    plt.plot(np.linspace(0,2500,200),J, lw = 3, c = 'r')
    plt.savefig('images/spectralDen.png')