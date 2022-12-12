# This is a short bash script to help setting up some things
# Install some python packages:
pip install numpy scipy matplotlib classy h5py
pip install mpi4py # This may not work in some cases, that is ok!

# Create some directories
mkdir -p notebooks/img
mkdir -p caches

# Download the IGRB table
echo  'If you want to use all aspects of gammaray module, you need to download
       the isotropic gamma ray background table from
       https://iopscience.iop.org/article/10.1088/0004-637X/799/1/86#apj504089t3
       (scrolling a little bit down and "Download table as: Data") and place it 
       under the filename "data/apj504089t3_mrt.txt
       (Sorry, I did not know how to automatize this due to access tokens)
       However, you can skip this step for now and do it later, when you notice that you need it'

# download precomputed caches
# You may skip this, if you want to recompute those yourself
pip install gdown  # This program allows to download files from google drive easily
gdown 1-KVfrJ_D1452hyqFBmJCFSDdyvqHwbHw
# If you don't want to use this program, you can also download the file manually from here:
# https://drive.google.com/file/d/1-KVfrJ_D1452hyqFBmJCFSDdyvqHwbHw/view?usp=share_link

tar -zxvf caches.tar.gz -C .

rm caches.tar.gz
