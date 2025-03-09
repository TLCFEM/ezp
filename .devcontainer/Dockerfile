FROM fedora:41

RUN echo -e "[oneAPI]\nname=Intel® oneAPI repository\nbaseurl=https://yum.repos.intel.com/oneapi\nenabled=1\ngpgcheck=1\nrepo_gpgcheck=1\ngpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB" > /etc/yum.repos.d/oneAPI.repo

RUN dnf install -y intel-oneapi-mkl-devel intel-oneapi-mpi-devel cmake gcc g++ gfortran git ninja-build
