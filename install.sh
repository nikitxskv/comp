sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python-dev python-pip python-nose g++ libblas-dev git cmake gfortran liblapack-dev zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig clang unzip htop python-setuptools libibnetdisc-dev
sudo pip install -U pip
sudo pip install -U numpy pandas scipy matplotlib sklearn jupyter seaborn tqdm scikit-image hyperopt
sudo pip uninstall -y hyperopt
sudo pip install hyperopt==0.0.2

# XGBoost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd
cd xgboost; cd python-package; sudo python setup.py install
cd
echo export PYTHONPATH=~/xgboost/python-package >> .bashrc

# OpenMpi
wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.0.tar.gz
tar -xvzf openmpi-2.1.0.tar.gz
cd openmpi-2.1.0
./configure --prefix="/home/$USER/.openmpi"
make && sudo make install
export PATH="$PATH:/home/$USER/.openmpi/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.openmpi/lib/"

echo export PATH="$PATH:/home/$USER/.openmpi/bin" >> .bashrc
echo export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.openmpi/lib/" >> .bashrc
cd
rm openmpi-2.1.0.tar.gz

# LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake -DUSE_MPI=ON ..
make -j
cd
cd LightGBM/python-package/
sudo python setup.py install
cd