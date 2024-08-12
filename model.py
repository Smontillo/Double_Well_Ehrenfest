import numpy as np
import numba as nb
import parameters as par
# ===========================

# ELECTRONIC - BATH COUPLING HAMILTONIAN
@nb.jit(nopython=True, fastmath=True)
def H_BC(data):
    H0 = np.zeros((par.nDW,par.nDW), dtype = np.complex128)
    H0  -= np.sum(par.cj * data.x) * par.R 
    data.H_bc = H0 * 1.0

# INITIALIZE BATH DOF
@nb.jit(nopython=True, fastmath=True)
def initR(data):
    β  = par.β
    ωj = par.ωj

    # POSITION AND MOMENTA ARE SAMPLE FROM A GAUSSIAN DISTRIBUTION WITH STANDARD DEVIATION σx AND σP.
    σP = np.sqrt(ωj / (2 * np.tanh(0.5*β*ωj)))
    σx = σP/ωj

    data.x = np.random.normal(loc=0.0, scale=1.0, size= len(ωj)) * σx
    data.P = np.random.normal(loc=0.0, scale=1.0, size= len(ωj)) * σP