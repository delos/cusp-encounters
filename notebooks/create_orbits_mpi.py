# You can use this file to create orbits of the MilkyWay in parallel
# The results are automatically cached into ../caches/milkyway_orbits.hdf5
# To use with mpi parallelization on e.g. 8 processors:
# mpirun -n 8 python -u create_orbits_mpi.py
# With the current parameter setup this will take a couple of hours. To
# first test that everything is working, you might wanna reduce e.g.
# the number of integration steps below, before executing at full length
import numpy as np
import time

import sys
sys.path.append("../adiabatic-tides")
sys.path.append("..")
import cusp_encounters.milkyway

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

cachedir = "../caches"
mw = cusp_encounters.milkyway.MilkyWay(adiabatic_contraction=True, cachedir=cachedir, mode="cautun_2020")

t0 = time.time()
orbits = mw.create_dm_orbits(100000, nsteps=100000, rmax=500e3, addinfo=True, adaptive=True, subsamp=100, mpicomm=comm) # 500
print("Task %d took %.1f seconds" % (rank, time.time() - t0))

if rank == 0:
    for key in orbits:
        print("Shape of block ", key, " is ", orbits[key].shape)
