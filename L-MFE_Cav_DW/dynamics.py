#!/software/anaconda3/2020.11/bin/python
#SBATCH -p debug
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=qjob.out
#SBATCH --error=qjob.err
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import numpy as np
import time as tm
import sys, os
import MFE as method
import parameters as par
import TrajClass as tc
import model
# =================================
# =========================
# Parallelization
# =========================
# RUN PARALLEL TRAJECTORIES
# THE NUMBER OF TRAJECTORIES PER JOB (j) IS DETERMINED BASED ON THE NUMBER OF CPUS (par.Cpus) AND TOTAL TRAJECTORIES (par.NTraj)
# j = NTraj / Cpus
parallel = par.parallel

if (parallel == True):
    sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # INCLUDE PARENT DIRECTORY WHICH HAS METHOD AND MODEL FILES
    JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"])               # GET ID OF THIS JOB
    TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"])             # GET ID OF THIS TASK WITHIN THE ARRAY 

    nrank = int(TASKID)                                         # JOD ID FOR A JOB 
    size  = par.Cpus                                            # TOTAL NUMBER OF PROCESSOR AVAILABLE
else:
    nrank = 0
    size  = 1

# =================================
# COMPILATION 
# =================================
# WITH JIT, THE CODE MUST BE COMPILE FIRST. RUN THE CODE FOR ONLY TWO TIME STEPS FIRST
com_ti = tm.time()
nDW_dummy = par.nDW
nc_dummy = par.nc
ndof_dummy = par.ndof
nsteps_dummy = 2
data_dummy = tc.trajData(nDW_dummy, nc_dummy, ndof_dummy, nsteps_dummy, nsteps_dummy)
data_dummy.ψt = par.ψ0

# MODEL FUNCTIONS =================
model.initR(data_dummy)
model.H_BC(data_dummy)

# METHOD FUNCTIONS ================
method.Force1(data_dummy)
method.RK4(data_dummy)
method.VelVer(data_dummy)
method.run_traj(data_dummy)

com_tf = tm.time()
print(f'Compilation time --> {np.round(com_tf - com_ti,2)} s or {np.round((com_tf - com_ti)/60,2)} min')


# =================================
# SIMULATION
# =================================
# DIVIDE THE NUMBER OF TRAJECTORIES PER JOB BASE ON THE NUMBER OF PROCESSORS AND TOTAL TRAJECTORIES
tot_Tasks = par.NTraj
NTasks = tot_Tasks//size
NRem = tot_Tasks - (NTasks*size)
TaskArray = [i for i in range(nrank * NTasks , (nrank+1) * NTasks)]
for i in range(NRem):
    if i == nrank: 
        TaskArray.append((NTasks*size)+i)
TaskArray = np.array(TaskArray)                                  # CONTAINS THE NUMBER OF TRAJECTORIES ASSIGNED TO EACH JOB
# =================================

ψw = np.zeros((par.nData, par.nt))                              # DENSITY MATRIX AVERAGED OVER THE NUMBER OF TRAJECTORIES ASSIGNED TO THIS JOB
trajData = tc.trajData(par.nDW, par.nc, par.ndof, par.NSteps, par.nData) # INITIATE THE TIME DEPENDENT DATA

sim_ti = tm.time()
for i in range(len(TaskArray)):
    method.run_traj(trajData)
    ψw += trajData.ψw
sim_tf = tm.time()
t = (sim_tf - sim_ti)
times = [t, t/60, t/(60*60)]
time_tj = (sim_tf - sim_ti)/len(TaskArray)
print(f'Simulation time --> {np.round(times[0],2)} s, {np.round(times[1],2)} min, {np.round(times[2],2)} h')
print(f'Time per trajectory -> {np.round(time_tj,2)} s')
print(' ================================================================================================= ')

try:
    np.savetxt(f'../data/rho_{nrank}.txt', ψw/len(TaskArray))   # RUN IN PARALLEL
except:
    np.savetxt(f'./data/rho_{nrank}.txt', ψw/len(TaskArray))    # RUN IN SERIES
