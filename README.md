Video
https://youtu.be/2tRjzgLpg_Q

# cusp-encounters
Code for evaluating the effect of stellar encounters on the annihilation luminosities of prompt cusps

clone with
git clone --recurse-submodules git@github.com:jstuecker/cusp-encounters.git

or if you forgot the submodule part you can do after a normal clone:
git submodule init
git submodule update

Done:
notebooks/milkyway_orbits.py    (still have to remove the cusp-cusp plots)
notebooks/annihilation_results.ipynb

notebooks/create_orbits_mpi.py
notebooks/gammaray.py


References 
CLASS http://www.class-code.net/

https://github.com/delos/microhalo-models (Delos et al, 2019)


Requirements
pip install numpy, scipy, classy