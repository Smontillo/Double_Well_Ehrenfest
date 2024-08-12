import numpy as np
import numba as nb
import model
import parameters as par
# ==================================

# FORCE MATRIX
@nb.jit(nopython=True, fastmath=True)
def Force(data, F):
    F[:]    = 0
    F[:]   -= par.ωj**2 * data.x * 1.0
    par_sum = np.sum(par.dHij * 1.0 * data.ρt, axis= 1).real    # NUMBA DOES NOT ALLOW THE SUMMATION OVER TWO AXIS, MUST BE DONE IN TWO STEPS
    F[:]   -= np.sum(par_sum, axis = 1).real

# RUNGE - KUTTA PROPAGATOR
# THIS DOES HALF OF THE ELETRONIC STEPS, par.Estep/2 | Estep MUST BE EVEN!!!
@nb.jit(nopython=True, fastmath=True)
def RK4(data):
    H  = par.H0 * 1.0 + data.H_bc * 1.0
    ρ  = data.ρt * 1.0
    dt = par.dtE * 1.0
    for k in range(int(par.Estep/2)):
        k1 = von_Newman(ρ.copy(), H.copy())
        k2 = von_Newman(ρ.copy() + 0.5 * dt * k1, H.copy())
        k3 = von_Newman(ρ.copy() + 0.5 * dt * k2, H.copy())
        k4 = von_Newman(ρ.copy() + dt * k3, H.copy())
        ρ  = ρ.copy() + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    data.ρt = 1.0 * ρ

# VON - NEWMAN EQUATION
@nb.jit(nopython=True, fastmath=True)
def von_Newman(ρf, H):
    return -1j * (H @ ρf - ρf @ H)

#  VELOCITY VERLET PROPAGATOR
@nb.jit(nopython=True, fastmath=True)
def VelVer(data) : 
    data.v[:] = data.P[:]/par.M                                                 # VELOCITY           
    RK4(data)                                                                   # ELECTRONIC UPDATE | RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!
    # ======= Nuclear Block ==================================
    data.x[:] += data.v[:] * par.dtN + 0.5 * data.F1[:] * par.dtN** 2 / par.M   # POSITION UPDATE
    model.H_BC(data)                                                            # ELECTRONIC HAMILTONIAN UPDATE | BATH POSITION DEPEDENT PART (CHECK model.py FILE) 
    #-----------------------------
    Force(data, data.F2)                                                        # FORCE AT t2
    data.v[:] += 0.5 * (data.F1[:] + data.F2[:]) * par.dtN / par.M              # VELOCITY UPDATE
    data.P[:] = data.v[:] * par.M                                               # MOMENTUM UPDATE
    data.F1[:] = data.F2[:] * 1.0                                               # SET F1 AS F2 FOR THE NEXT STEP
    # ======================================================
    RK4(data)                                                                   # ELECTRONIC UPDATE | RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!

# RUN TRAJECTORIES
@nb.jit(nopython=True, fastmath=True)
def run_traj(data):
    model.initR(data)                           # INITIALIZE x AND P FOR ALL BATH MODES
    data.ρt = par.ρ0                            # INITIAL DENSITY MATRTIX
    Force(data, data.F1)                        # FORCE AT t = 0

    iskip = 0
    for st in range(data.nsteps):
        if (st % par.nskip == 0):               # WRITTING OF THE DENSITY MATRIX
            ρ = data.ρt
            ρ = ρ.copy()
            data.ρw[iskip,:]  = ρ.reshape(1,par.nDW**2)
            iskip += 1

        VelVer(data)                            # EVOLUTION OF THE SYSTEM FOR nsteps
        