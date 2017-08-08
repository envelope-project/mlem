# MLEM - Maximum Liklihood Expectation Maximization for Small PET Devices

The efficient use of multicore architectures for sparse matrix-vector multiplication (SpMV) is currently an open challenge. One algorithm which makes use of SpMV is the maximum likelihood expectation maximization (MLEM) algorithm. When using MLEM for positron emission tomography (PET) image reconstruction, one requires a particularly large matrix. We present a new storage scheme for this type of matrix which cuts the memory requirements by half, compared to the widely-used compressed sparse row format. For parallelization we combine the two partitioning techniques recursive bisection and striping. Our results show good load balancing and cache behavior. We also give speedup measurements on various modern multicore systems.

## Authors
- Initial Verison and Algorithmic, MPI Version: Tilman Kuestner
- LAIK Version, HPC Optimization: Josef Weidendorfer
- LAIK Version, Measurements, HPC Optimizations: Dai Yang
- Generators: Thorsten Fuchs

## Dependencies
- [libb0ost 1.58](http://boost.org/), for the iterators and program options.
- C++ 11, GNU Compiler
- OpenMP Support
- libconfig++
- [LAIK Library](https://github.com/envelope-project/laik)

## Usage
Different versions are supplied within this repository. 
- mpicsr4mlem: the native MPI Version without LAIK support.
- laikcsr4mlem: the LAIK version without repartitioning support. 
- laikcsr4mlem-repart: LAIK version with explicit repartitioning. 
- csr4gen/csr4gen: The CSR (Compressed Sparse Rows) matrix generator. 
- singen/singen: The sinogram (image raw data) generator. 


### Compile
```sh
 make
```
### Generate a Sparse Matrix for Testing
```sh
./csr4gen/csr4gen -c ./examples/madpet2.cfg -d 0.01 -v 0.1 -f test.csr4
./singen/singen 65536
```
### Run the nativ MPI Version
```sh
mpirun -np <num_tasks> ./mpicsr4mlem <matrix> <input> <output> <iterations> <checkpointing>
```
### native mpi example
```sh
mpirun -np 4 ./mpicsr4mlem test.csr4 sino65536.sino mlem-60.out 60 0
```
### Run the LAIK Version
```sh
mpirun -np <num_tasks> ./laikcsr4mlem <matrix> <input> <output> <iterations>
```
### LAIK example
```sh
mpirun -np 4 ./laikcsr4mlem test.csr4 sino65536.sino mlem-60.out 60 0
```

## Build Flags:
- -D\_HPC\_ enables the HPC optimization of this code. It uses memcpy instead of mmap to optimize NUMA operations. 
- -DMESSUNG disables default outputs and enables output for measurement for clusters. 

## Cite
Any publication using the MLEM code must be informed to the authors of this code, e.g. Tilman Küstner.
- Original MLEM paper: L. A. Shepp and Y. Vardi, "Maximum Likelihood Reconstruction for Emission Tomography," in IEEE Transactions on Medical Imaging, vol. 1, no. 2, pp. 113-122, Oct. 1982. doi: 10.1109/TMI.1982.4307558 [Link](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4307558&isnumber=4307552)
- "Parallel MLEM on Multicore Architectures", Tilman Küstner, Josef Weidendorfer, Jasmine Schirmer, Tobias Klug, Carsten Trinitis, Sibylle I. Ziegler. In: Computational Science - ICCS 2009, LNCS, vol. 5544, pp. 491-500, Springer, 2009. [Link](http://www.springerlink.com/content/x2226771p5779h34/)

## License
This code is distributed as a open source project under GPL License Version 3. Please refer to LICENSE document.

## Acknowledgement
This project is partially financed by project ENVELOPE, which is supported by Federal Ministry of Education and Research (BMBF) of the Federal Republic of Germany. 