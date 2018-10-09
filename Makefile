CXX     = g++
CC		= gcc
MPICXX	= mpicxx
mpicc 	= mpicc

RM      = rm -f

#-include Makefile.config

CXXFLAGS  = -Wall -std=c++11 -O3 -fstrict-aliasing -L/usr/local/bin
LDFLAGS  = -Wall -O3
SOURCES = src/csr4matrix.cpp src/scannerconfig.cpp
HEADERS = $(wildcard include/*.hpp)
OBJECTS = $(SOURCES:%.cpp=%.o)

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
PIN_CFLAGS = -D_PINNING_

# GGF. XEON-PHI
KNL_CFLAGS = #-DXEON_PHI 
KNL_LFLAGS = #-lmemkind

#include 
CXXFLAGS += -I./include

all: mpicsr4mlem mpicsr4mlem2 openmpcsr4mlem laikcsr4mlem csr4gen singen

mpicsr4mlem: mpicsr4mlem.o $(OBJECTS)
	$(MPICXX) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem.o $(OBJECTS)
mpicsr4mlem.o: src/mpicsr4mlem.cpp
	$(MPICXX) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ -c $<
mpicsr4mlem2: mpicsr4mlem2.o $(OBJECTS)
	$(MPICXX) $(LFLAGS)  $(KNL_LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem2.o $(OBJECTS)
mpicsr4mlem2.o: src/mpicsr4mlem2.cpp
	$(MPICXX) $(KNL_CFLAGS) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ -c $<

laikcsr4mlem: src/laikcsr4mlem.cpp $(OBJECTS)
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

%.o: src/%.cpp
	$(CC) $(CXXFLAGS) $(DEFS) -o $@ -c $< 

csr4gen: 
	+cd csrgen && make

singen:
	+cd singen && make

clean:
	$(RM) *.o mpicsr4mlem laikcsr4mlem laikcsr4mlem-repart
	cd csrgen && make clean
	cd singen && make clean

