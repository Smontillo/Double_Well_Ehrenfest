import numpy as np
from numba import int32, float64, complex128
from numba.experimental import jitclass
from numba import jit
import parameters as par
# ==================================

spec = [
    ('nt',                      int32), # NUMBER OF VIBRATIONAL STATES 
    ('nsteps',                  int32), # NUMBER OF EVOLUTION STEPS
    ('ndof',                    int32), # NUMBER OF BATH MODES
    ('x',                  float64[:]), # BATH POSITION
    ('P',                  float64[:]), # BATH MOMENTA
    ('v',                  float64[:]), # BATH VELOCITY
    ('F1',                 float64[:]), # BATH FORCE AT t
    ('F2',                 float64[:]), # BATH FORCE AT t + 1
    ('ρt',            complex128[:,:]), # DENSITY MATRIX AT TIME t
    ('H_bc',          complex128[:,:]), # ELECTRONIC HAMILTONIAN | DEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
    ('ρw',            complex128[:,:]), # PLACE HOLDER FOR THE DENSITY MATRIX
]

@jitclass(spec)
class trajData(object):
    def __init__(self, nDW, ndof, nsteps):
        self.nt     = nDW
        self.nsteps = nsteps
        self.ndof   = ndof
        self.x      = np.zeros(self.nodf, dtype = np.float64)
        self.P      = np.zeros(self.nodf, dtype = np.float64)
        self.v      = np.zeros(self.nodf, dtype = np.float64)
        self.F1     = np.zeros(self.nodf, dtype = np.float64)
        self.F2     = np.zeros(self.nodf, dtype = np.float64)
        self.ρt     = np.zeros((self.nt, self.nt), dtype = np.complex128)
        self.H_bc   = np.zeros((self.nt, self.nt), dtype = np.complex128)
        self.ρw     = np.zeros((nsteps,self.nt**2) , dtype = np.complex128)