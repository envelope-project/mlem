# MLEM - Maximum Liklihood Expectation Maximization for Small PET Devices

## Dependencies
- [libbost 1.58] (http://boost.org/)
- C++ 11, GNU Compiler
- OpenMP Support
- [LAIK Library](https://github.com/envelope-project/laik)

### Compile
```sh
$ make
```

### Run
```sh
./mpicsr4mlem <matrix> <input> <output> <iterations> <checkpointing>
```
### for example
```sh
mpirun -np 4 ./mpicsr4mlem madpet2.csr4 test1.LMsino mlem-60.out 60 0
```

## Cite
Any publication using the MLEM code must be informed to the authors of this code, e.g. Tilman Küstner.
** THE ORIGINAL MLEM PAPER MUST BE CITED. **

https://link.springer.com/chapter/10.1007%2F978-3-642-01970-8_48
