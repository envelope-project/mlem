CC      = icpc
RM      = rm -f

#-include Makefile.config

CXXFLAGS  = -Wall -std=c++11 -O3 -fstrict-aliasing
LFLAGS  = -Wall -O3
OMP_FLAGS = -fopenmp

SOURCES = $(wildcard *.cc)
OBJECTS = $(SRC_GEN:%.cc=%.o)

#Libboost
CXXFLAGS += $(BOOST_INC) -lboost_program_options

#include 
CXXFLAGS += -I./include

all: objs csr4gen sinogen

csr4gen: main.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ main.o $(OBJECTS)

sinogen: sinogen.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ sinogen.o $(OBJECTS) 
	
objs: $(SOURCES)
	$(CC) $(CXXFLAGS) $(OMP_FLAGS) $(DEFS) -c $^ $< 

clean:
	- $(RM) *.o csr4gen sinogen

