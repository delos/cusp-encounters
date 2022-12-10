# This is a short bash script to help setting up some things
# Install some python packages:
pip install numpy scipy matplotlib classy h5py
pip install mpi4py # This may not work in some cases, that is ok!

# Create some directories
mkdir -p notebooks/img
mkdir -p caches

# Download the IGRB table
# If this link does not work: You can find this table as the IGRB table published wit the paper Ackermann et al. (2015)
wget -O data/igrb_ackermann_2015.txt "https://cfn-live-content-bucket-iop-org.s3.amazonaws.com/journals/0004-637X/799/1/86/revision1/apj504089t3_mrt.txt?AWSAccessKeyId=AKIAYDKQL6LTV7YY2HIK&Expires=1670690788&Signature=%2Fzs%2BfilOfCNJhaaVxBu01EFnzHI%3D"

# download precomputed caches
# You may skip this, if you want to recompute those yourself
pip install gdown  # This program allows to download files from google drive easily
gdown 1-KVfrJ_D1452hyqFBmJCFSDdyvqHwbHw
# If you don't want to use this program, you can also download the file manually from here:
# https://drive.google.com/file/d/1-KVfrJ_D1452hyqFBmJCFSDdyvqHwbHw/view?usp=share_link

tar -zxvf caches.tar.gz -C .

rm caches.tar.gz
