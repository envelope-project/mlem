CXX     = g++
CC		= gcc
MPICXX	= mpicxx
mpicc 	= mpicc
NVCC	= nvcc

RM      = rm -f

#-include Makefile.config

CXXFLAGS	= -std=c++11 -O3
LDFLAGS		= -O3
SOURCES		= src/csr4matrix.cpp src/scannerconfig.cpp
HEADERS		= $(wildcard include/*.hpp)
OBJECTS 	= $(SOURCES:%.cpp=%.o)
CULDFLAGS 	= -lcublas -lcusparse -lnvidia-ml -lnccl
CUCXXFLAGS	= -Xcompiler

MESSUNG_FLAG = #-DMESSUNG
CXXFLAGS += $(MESSUNG_FLAG)

# LAIK lib
LAIK_ROOT = ./laik
LAIK_INC =-I$(LAIK_ROOT)/include/
LAIK_LIB =-L$(LAIK_ROOT)/ -llaik

#Libboost
CXXFLAGS += $(BOOST_INC)

#openmp
OMP_FLAGS = -fopenmp
PIN_CFLAGS = -D_PINNING_ -D_OMP_

# GGF. XEON-PHI
KNL_CFLAGS = #-DXEON_PHI 
KNL_LFLAGS = #-lmemkind

#include 
CXXFLAGS += -I./include

all: mpicsr4mlem mpicsr4mlem2 mpicsr4mlem3 openmpcsr4mlem laikcsr4mlem cudacsr4mlem

mpicsr4mlem: mpicsr4mlem.o $(OBJECTS)
	$(MPICXX) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem.o $(OBJECTS)
mpicsr4mlem.o: src/mpicsr4mlem.cpp
	$(MPICXX) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ -c $<
mpicsr4mlem2: mpicsr4mlem2.o $(OBJECTS)
	$(MPICXX) $(LFLAGS)  $(KNL_LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem2.o $(OBJECTS)
mpicsr4mlem2.o: src/mpicsr4mlem2.cpp
	$(MPICXX) $(KNL_CFLAGS) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ -c $<

# "LAIK" Version 
laikcsr4mlem: src/laikcsr4mlem.cpp $(OBJECTS)
	$(MPICXX) -g -O3 $(CXXFLAGS) $< $(LAIK_INC) -Wall -DUSE_MPI=1 -fopenmp -Wl,-rpath,$(abspath $(LAIK_ROOT)) $(LAIK_LIB) $(OBJECTS) -o $@
laikcsr4mlem-repart: src/laikcsr4mlem-repart.cpp $(OBJECTS)
	$(MPICXX) -g -O3 $(CXXFLAGS) $< $(LAIK_INC) -Wall -DUSE_MPI=1 -fopenmp -Wl,-rpath,$(abspath $(LAIK_ROOT)) $(LAIK_LIB) $(OBJECTS) -o $@

#PURE OpenMP Version
openmpcsr4mlem: openmpcsr4mlem.o $(OBJECTS)
	$(CXX) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ openmpcsr4mlem.o $(OBJECTS)
openmpcsr4mlem.o: src/openmpcsr4mlem.cpp
	$(CXX) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ -c $<

openmpcsr4mlem-pin: openmpcsr4mlem-pin.o $(OBJECTS)
	$(CXX) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ openmpcsr4mlem-pin.o $(OBJECTS)
openmpcsr4mlem-pin.o: src/openmpcsr4mlem.cpp
	$(CXX) $(CXXFLAGS) $(DEFS) $(PIN_CFLAGS) $(OMP_FLAGS) -o $@ -c $<

openmpcsr4mlem-knl: openmpcsr4mlem-knl.o $(OBJECTS)
	$(CXX) $(LFLAGS) $(KNL_LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ openmpcsr4mlem-knl.o $(OBJECTS)
openmpcsr4mlem-knl.o: src/openmpcsr4mlem.cpp
	$(CXX) $(CXXFLAGS) $(DEFS) $(KNL_CFLAGS) $(OMP_FLAGS) -o $@ -c $<

# Hybrid Version of MLEM
fusedmpimlem: fusedmpimlem.o profiling.o $(OBJECTS)
	$(MPICXX) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ fusedmpimlem.o profiling.o $(OBJECTS)
fusedmpimlem.o: src/fusedmpimlem.cpp
	$(MPICXX) $(OMP_FLAGS) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ -c $<
mpicsr4mlem3: mpicsr4mlem3.o $(OBJECTS)
	$(MPICXX) $(LFLAGS) $(DEFS) $(OMP_FLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem3.o $(OBJECTS)
mpicsr4mlem3.o: src/mpicsr4mlem3.cpp
	$(MPICXX) $(OMP_FLAGS) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ -c $<


# CUDA GPU Version
cudacsr4mlem: cudacsr4mlem.o cudacsr4mlemkernel.o $(OBJECTS)
	$(NVCC) $(CUCXXFLAGS) $(OMP_FLAGS) $(CULDFLAGS) $(LFLAGS) $(DEFS) -o $@ cudacsr4mlem.o cudacsr4mlemkernel.o $(OBJECTS) $(CUDA_OBJECTS) 
cudacsr4mlem.o: src/cudacsr4mlem.cu
	$(NVCC) $(CUCXXFLAGS) $(OMP_FLAGS) $(CXXFLAGS) $(CFLAGS) $(DEFS) -I./include -o $@  -c $<
cudacsr4mlemkernel.o: src/cudacsr4mlemkernel.cu
	$(NVCC) $(CUCXXFLAGS) $(OMP_FLAGS) $(CXXFLAGS) $(CFLAGS) $(DEFS) -I./include -o $@ -c $<



#objects
%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c $< 
profiling.o: src/profiling.c
	$(CC) $(CFLAGS) $(DEFS) -o $@ -c $< 

clean:
	$(RM) src/*.o
	$(RM) *.o *csr4mlem*
	$(RM) -r *.dSYM

