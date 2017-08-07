CC      = mpicxx
RM      = rm -f

#-include Makefile.config

CXXFLAGS  = -Wall -std=c++11 -O3 -fstrict-aliasing -L/usr/local/bin
LDFLAGS  = -Wall -O3
OMP_FLAGS = -fopenmp
SOURCES = src/csr4matrix.cpp src/scannerconfig.cpp
HEADERS = $(wildcard include/*.hpp)
OBJECTS = $(SOURCES:%.cc=%.o)

# LAIK lib
LDFLAGS += -llaik

#Libboost
CXXFLAGS += $(BOOST_INC)

#openmp
CXXFLAGS += $(OMP_FLAGS)

#include 
CXXFLAGS += -I./include

all: mpicsr4mlem laikcsr4mlem laikcsr4mlem-repart csr4gen singen

mpicsr4mlem: mpicsr4mlem.o $(OBJECTS)
	$(CC) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem.o $(OBJECTS) $(LDFLAGS)

laikcsr4mlem: laikcsr4mlem.o $(OBJECTS)
	$(CC)  $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ laikcsr4mlem.o $(OBJECTS) $(LDFLAGS)

laikcsr4mlem-repart: laikcsr4mlem-repart.o $(OBJECTS)
	$(CC) $(CXXFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ laikcsr4mlem-repart.o $(OBJECTS) $(LDFLAGS)

%.o: src/%.cpp
	$(CC) $(CXXFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $< 

csr4gen: 
	+cd csrgen && make

singen:
	+cd singen && make

clean:
	$(RM) *.o mpicsr4mlem laikcsr4mlem laikcsr4mlem-repart
	cd csrgen && make clean
	cd singen && make clean

