import numpy as np
import numba as nb
import model
import parameters as par
# ==================================
# LINDBLAD UPDATE
@nb.jit(nopython=True, fastmath=True)
def LindLoss(data, proc):
    # THE DECAY IS TAKEN FROM STATE 0 TO STATE 1
    # FOR THE CAVITY, TWO DISSIPATION CHANNELS ARE USED
    #  Exc ⇒ EXCITATION       Rel → RELAXATION
    if proc == 'Exc':
        a, b, Γ = 0, 1, par.Γexc
    elif proc == 'Rel':
        a, b, Γ = 1, 0, par.Γrel
    else:
        print('Incorrect dissipation channel')
    ψ = data.ψt * 1.0
    L = par.LID 
    dt = par.dtN/4
    # Initial Coeff
    c0 = ψ[a * par.nDW: (a + 1) * par.nDW]
    c1 = ψ[b * par.nDW: (b + 1) * par.nDW]
    # Tot. Pop.
    pop0, pop1 = np.abs(c0*np.conj(c0)), np.abs(c1*np.conj(c1))
    # Propagated Coeff.
    c0P, c1P = np.exp(- Γ * dt/2) * c0, np.sqrt(pop1 + pop0 * (1 - np.exp(- Γ * dt))) + 0j
    Θ = np.zeros((par.nDW))
    for k in range(par.nDW):
        if(np.real(c1P[k]) > 0):
            # Determine off-diagonal decay rate
                dr = np.real(np.abs(c1[k]) / c1P[k])
            # Use spline interpolation to find dΘ
                if(dr<1.0): # dr is the a in sin(x) = ax
                    drnum = dr*1000000
                    dΘ = L[int(np.floor(drnum)),1]*(1+np.floor(drnum)-drnum) + L[int(np.floor(drnum))+1,1]*(drnum-np.floor(drnum))
                else:
                    dΘ = 0.0
    # Calculate Θ from uniform distribution [-dΘ,dΘ]
        Θ[k] = np.random.uniform(-dΘ,dΘ)
    # Propogate random phase of coefficients
    c1P *= np.exp(1j * (Θ + np.angle(c1))) + 0j
    ψ[a * par.nDW: (a + 1) * par.nDW], ψ[b * par.nDW: (b + 1) * par.nDW] = c0P, c1P
    data.ψt = ψ * 1.0

# FORCE 1 UPDATE FUNCTION
@nb.jit(nopython=True, fastmath=True)
def Force1(data):
    data.F1[:]    = 0
    data.F1[:]   -= par.ωj[:]**2 * data.x[:] 
    ψ = data.ψt
    out = np.outer(ψ.conjugate(),ψ)
    par_sum  = np.sum(par.dHij * out * 1.0, axis= 1).real    # NUMBA DOES NOT ALLOW THE SUMMATION OVER TWO AXIS, MUST BE DONE IN TWO STEPS
    data.F1[:]   -= np.sum(par_sum, axis = 1).real

# FORCE 2 UPDATE FUNCTION
@nb.jit(nopython=True, fastmath=True)
def Force2(data):
    data.F2[:]    = 0
    data.F2[:]   -= par.ωj[:]**2 * data.x[:] 
    ψ = data.ψt
    out = np.outer(ψ.conjugate(),ψ)
    par_sum  = np.sum(par.dHij * out * 1.0, axis= 1).real     # NUMBA DOES NOT ALLOW THE SUMMATION OVER TWO AXIS, MUST BE DONE IN TWO STEPS
    data.F2[:]   -= np.sum(par_sum, axis = 1).real

# RUNGE - KUTTA PROPAGATOR
# THIS DOES HALF OF THE ELETRONIC STEPS, par.Estep/2 | Estep MUST BE EVEN!!!
@nb.jit(nopython=True, fastmath=True)
def RK4(data):
    H  = par.Hel * 1.0 + data.H_bc * 1.0
    ψ  = data.ψt * 1.0
    dt = par.dtE * 1.0
    for k in range(int(par.Estep/2)):
        q, p = np.sqrt(2)*np.real(ψ).astype(np.complex_), np.sqrt(2)*np.imag(ψ).astype(np.complex_)
        p -= 0.5 * dt * H @ q
        q += dt * H @ p
        p -= 0.5 * dt * H @ q
        ψ = (q+1j*p)/np.sqrt(2)
    data.ψt = 1.0 * ψ

# @nb.jit(nopython=True, fastmath=True)
# def RK4(data):
#     data.H[:,:]  = par.Hel * 1.0 + data.H_bc * 1.0
#     data.ψ[:]  = data.ψt * 1.0
#     dt = par.dtE * 1.0
#     for k in range(int(par.Estep/2)):
#         data.q[:], data.p[:] = np.sqrt(2)*np.real(data.ψ).astype(np.complex128), np.sqrt(2)*np.imag(data.ψ).astype(np.complex128)
#         data.p[:] -= 0.5 * dt * data.H @ data.q
#         data.q[:] += dt * data.H @ data.p
#         data.p[:] -= 0.5 * dt * data.H @ data.q
#         data.ψ[:] = (data.q+1j*data.p)/np.sqrt(2)
#     data.ψt[:] = 1.0 * data.ψ

#  VELOCITY VERLET PROPAGATOR
@nb.jit(nopython=True, fastmath=True)
def VelVer(data) : 
    data.v[:] = data.P[:]/par.M * 1.0                                           # VELOCITY           
    LindLoss(data, 'Exc')
    LindLoss(data, 'Rel')
    RK4(data)                                                                   # ELECTRONIC UPDATE | RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!
    LindLoss(data, 'Exc')
    LindLoss(data, 'Rel')
    data.x[:] += data.v[:] * par.dtN + 0.5 * data.F1[:] * par.dtN**2 / par.M    # POSITION UPDATE
    model.H_BC(data)                                                            # ELECTRONIC HAMILTONIAN UPDATE | BATH POSITION DEPEDENT PART (CHECK model.py FILE) 
    LindLoss(data, 'Exc')
    LindLoss(data, 'Rel')
    RK4(data)                                                                   # ELECTRONIC UPDATE | RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!
    LindLoss(data, 'Exc')
    LindLoss(data, 'Rel')
    Force2(data)                                                                # FORCE AT t2                                                   
    data.v[:] += 0.5 * (data.F1[:] + data.F2[:]) * par.dtN / par.M              # VELOCITY UPDATE
    data.P[:]  = data.v[:] * par.M * 1.0                                        # MOMENTUM UPDATE
    data.F1[:] = data.F2[:] * 1.0                                               # SET F1 AS F2 FOR THE NEXT STEP
    # ======================================================

# RUN TRAJECTORIES
@nb.jit(nopython=True, fastmath=True)
def run_traj(data):
    model.initR(data)                           # INITIALIZE x AND P FOR ALL BATH MODES
    data.ψt[:] = par.ψ0                            # INITIAL DENSITY MATRTIX
    Force1(data)                                # FORCE 1 AT t = 0
    model.H_BC(data)                            # INITIAL ELECTRONIC HAMILTONIAN UPDATE | BATH POSITION DEPEDENT PART 

    iskip = 0
    for st in range(data.nSteps):
        if (st % par.nskip == 0):               # WRITTING OF THE DENSITY MATRIX | SAVE ONLY THE DIAGONAL ELEMENTS
            ψ = data.ψt
            pop = np.conjugate(ψ) * ψ
            data.ψw[iskip,:] = np.real(pop)
            iskip += 1

        VelVer(data)                            # EVOLUTION OF THE SYSTEM FOR nsteps
        