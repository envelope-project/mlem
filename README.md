# MLEM - Maximum Liklihood Expectation Maximization for Small PET Devices

The efficient use of multicore architectures for sparse matrix-vector multiplication (SpMV) is currently an open challenge. One algorithm which makes use of SpMV is the maximum likelihood expectation maximization (MLEM) algorithm. When using MLEM for positron emission tomography (PET) image reconstruction, one requires a particularly large matrix. We present a new storage scheme for this type of matrix which cuts the memory requirements by half, compared to the widely-used compressed sparse row format. For parallelization we combine the two partitioning techniques recursive bisection and striping. Our results show good load balancing and cache behavior. We also give speedup measurements on various modern multicore systems.

## Authors
- Initial Verison and Algorithmic, MPI Version: Tilman Kuestner
- LAIK Version, HPC Optimization: Josef Weidendorfer, Tilman Kuestner
- LAIK Version, Measurements, HPC Optimizations: Dai Yang, Tilman Kuestner
- The new Hybrid and OpenMP version: Rami Al-Rihawi
- GPU version
- Generators: Thorsten Fuchs

## Artifacts and Dependencies
| Name        | MPI           | OpenMP  |  LAIK | Others|Description|
| ------------- |:-------------:|:-----:|:--:|:--:|:--|
| openmpcsr4mlem| no | yes | no | | Pure, native OpenMP Implementation.|
| openmpcsr4mlem-pin| no | yes | no | | NUMA - Optimized OpenMP Version using thread Pinning |
| openmpcsr4mlem-knl | no | yes | no | numactl, memkind | Special Optimized version for Intel© Xeon Phi© Knight's Landing (KNL) Processors |
| mpicsr4mlem     | yes | no | no | | Pure MPI Implementation | 
| mpicsr4mlem2 | yes | yes      |  no | | Hybrid MPI-OpenMP Implementation with Thread Pinning |
| laikcsr4mlem | yes  | no | yes |  | MLEM ported to LAIK to enable application-integrated Fault Tolerance. 


## Dependencies
- [Boost Program Options](http://boost.org/), for the iterators and program options.
- C++11 compatible, preferable GNU Compiler
- OpenMP Support
- libconfig++ (for the Matrix Generator only)
- [LAIK Library](https://github.com/envelope-project/laik) (for LAIK Version only)

## Usage
Different versions are supplied within this repository. 
- mpicsr4mlem: a hybrid MPI+OpenMP Version without LAIK support.
- laikcsr4mlem: the LAIK version without repartitioning support and optimized load balancing
- openmpcsr4mlem: a pure OpenMP Version of MLEM
- openmpcsr4mlem-pin: OpenMP explicit thread pinned version of MLEM
- openmpcsr4mlem-knl: Intel Knight's Landing Architecture optimized version of MLEM
- laikcsr4mlem-repart: LAIK version with explicit repartitioning. 
- csr4gen/csr4gen: The CSR (Compressed Sparse Rows) matrix generator. 
- singen/singen: The sinogram (image raw data) generator. 


### Compile
```sh
 make $TARGET
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
- "Enabling Application-Integrated Proactive Fault Tolerance", Yang, D., Weidendorfer, J., Trinitis, C., Küstner, T., & Ziegler, S. (2018). In  Parallel Computing is Everywhere 32 (pp. 475-484). [Link](https://books.google.de/books?id=ysFVDwAAQBAJ&lpg=PA475&ots=k57wIk8a4x&dq=Dai%20Yang%20laik&lr&pg=PA475#v=onepage&q&f=false)
- "Implementation and Evaluation of MLEM algorithm on Intel Xeon Phi Knights Landing (KNL) Processors", Rami Al-Rihawi, Master's Thesis (2018). [Link](https://mediatum.ub.tum.de/1455603)
- "Implementation and Evaluation of MLEM-Algorithm on GPU using CUDA", Apoorva Gupta, Master's Thesis (2018). [Link](https://mediatum.ub.tum.de/1443203)
- "Co-Scheduling in a Task-Based Programming Model", T. Becker, D. Yang, T. Kuestner, M. Schulz. In Proceedings of the 3rd Workshop on Co-Scheduling of HPC Applications (COSH 2018). DOI: 10.14459/2018md1428536.

## License
This code is distributed as a open source project under GPL License Version 3. Please refer to LICENSE document.

## Acknowledgement
This project is partially financed by project ENVELOPE, which is supported by Federal Ministry of Education and Research (BMBF) of the Federal Republic of Germany. 
Development of the KNL and testing is partially funded by the Liebniz Supercomputing Centre 
