# MLEM - Maximum Liklihood Expectation Maximization for Small PET Devices

The efficient use of multicore architectures for sparse matrix-vector multiplication (SpMV) is currently an open challenge. One algorithm which makes use of SpMV is the maximum likelihood expectation maximization (MLEM) algorithm. When using MLEM for positron emission tomography (PET) image reconstruction, one requires a particularly large matrix. We present a new storage scheme for this type of matrix which cuts the memory requirements by half, compared to the widely-used compressed sparse row format. For parallelization we combine the two partitioning techniques recursive bisection and striping. Our results show good load balancing and cache behavior. We also give speedup measurements on various modern multicore systems.

## Authors
- Initial Verison and Algorithmic, MPI Version: Tilman Kuestner
- LAIK Version, HPC Optimization: Josef Weidendorfer, Tilman Kuestner
- LAIK Version, Measurements, HPC Optimizations: Dai Yang, Tilman Kuestner
- The new Hybrid and OpenMP version: Rami Al-Rihawi, Tilman Kuestner, Dai Yang
- CUDA version: Apporva Gupta, Mengdi Wang, Dai Yang

## Artifacts and Dependencies
| Name        | MPI           | OpenMP  |  LAIK | Others|Description|
| ------------- |:-------------:|:-----:|:--:|:--:|:--|
| openmpcsr4mlem| no | yes | no | | Pure, native OpenMP Implementation.|
| openmpcsr4mlem-pin| no | yes | no | | NUMA - Optimized OpenMP Version using thread Pinning |
| openmpcsr4mlem-knl | no | yes | no | numactl, memkind | Special Optimized version for Intel© Xeon Phi© Knight's Landing (KNL) Processors |
| mpicsr4mlem     | yes | no | no | | Pure MPI Implementation | 
| mpicsr4mlem2 | yes | yes      |  no | | Hybrid MPI-OpenMP Implementation with Thread Pinning |
| mpicsr4mlem3 | yes | yes      |  no | | Hybrid MPI-OpenMP Implementation with Thread Pinning, HBM Optimization and Cache Blocking |
| cudacsr4mlem | no | yes      |  no | CUDA + NCCL| High Performance CUDA implementation with OpenMP acceleration
| laikcsr4mlem | yes  | no | yes |  | MLEM ported to LAIK to enable application-integrated Fault Tolerance. Tested with commit a96769f193b32ee6196e28a7c554259f9bd749ef of LAIK. |


## Dependencies
- [Boost Program Options](http://boost.org/), for the iterators and program options.
- C++11 compatible, preferable GNU Compiler
- OpenMP Support
- For LAIK version, in addition: [LAIK Library](https://github.com/envelope-project/laik) 
- For CUDA Version, in addition: [CUDA 10](https://developer.nvidia.com/cuda-toolkit), [NVIDIA Collective Communication Library](https://developer.nvidia.com/nccl), cuSparse, cuBlas, spTrans
- MLEM Data Sets, optainable at TUM University Library, t.b.d. 


### Compile
```sh
 make $TARGET
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
- "Co-Scheduling in a Task-Based Programming Model", T. Becker, D. Yang, T. Kuestner, M. Schulz. In Proceedings of the 3rd Workshop on Co-Scheduling of HPC Applications (COSH 2018). DOI: [10.14459/2018md1428536](https://mediatum.ub.tum.de/1428536).
- "Porting MLEM Algorithm for Heterogeneous Systems", Mengdi Wang, Bachelor's Thesis (2019). [Link](https://mediatum.ub.tum.de/1518886)
- " Exploring high bandwidth memory for PET Image Reconstruction", Dai Yang, Tilman Küstner, Rami Al-Rihawi, Martin Schulz (2019). In Parallel Computing (ParCo 19'). Accepted for Publication. [Link t.b.d.]()


## License and Legal
This code is distributed as a open source project under GPL License Version 3. Please refer to LICENSE document.

The CSR Matrix transposition routine ([spTrans](https://github.com/vtsynergy/sptrans)) in `cudacsr4mlem` is provided as an open-source software by Virginia Polytechnic Institute & State University (Virginia Tech) under the [GNU Lesser General Public License v2.1](https://github.com/vtsynergy/sptrans/blob/master/LICENSE).

## Acknowledgement 
This project is partially financed by project [ENVELOPE](http://envelope.itec.kit.edu), which is supported by Federal Ministry of Education and Research (BMBF) of the Federal Republic of Germany. 

Compute Resources for development and testing is partially sponsored by the Leibniz Supercomputing Centre ([LRZ](https://www.lrz.de)).

spTrans is provided by Wang et al., originally published as Parallel Transposition of Sparse Data Structures. Hao Wang, Weifeng Liu, Kaixi Hou, Wu-chun Feng. In Proceedings of the 30th International Conference on Supercomputing (ICS), Istanbul, Turkey, June 2016.