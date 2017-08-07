CC      = mpicxx
RM      = rm -f

#-include Makefile.config

CXXFLAGS  = -Wall -std=c++11 -O3 -fstrict-aliasing
LFLAGS  = -Wall -O3
OMP_FLAGS = -fopenmp

SOURCES = $(wildcard src/*.cpp)
HEADERS = $(wildcard include/*.hpp)

# LAIK lib
LAIKLIB = -llaik

#Libboost
CXXFLAGS += $(BOOST_INC)

#include 
CXXFLAGS += -I./include

all: objs genobjs mpicsr4mlem laikcsr4mlem laikcsr4mlem-repart csr4gen sinogen

mpicsr4mlem: mpicsr4mlem.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem.o $(OBJECTS)

laikcsr4mlem: laikcsr4mlem.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ laikcsr4mlem.o $(OBJECTS) $(LAIKLIB)

laikcsr4mlem-repart: laikcsr4mlem-repart.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ laikcsr4mlem-repart.o $(OBJECTS) $(LAIKLIB)

objs: $(SOURCES)
	$(CC) $(CXXFLAGS) $(OMP_FLAGS) $(DEFS) -c $^ $< 

clean:
	- $(RM) *.o mpicsr4mlem laikcsr4mlem

