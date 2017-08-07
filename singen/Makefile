CC      = g++
RM      = rm -f

#-include Makefile.config

CXXFLAGS  = -Wall -std=c++11 -O3 -fstrict-aliasing
LDFLAGS = 
OMP_FLAGS = -fopenmp


SOURCES = $(wildcard *.cc)
OBJECTS = $(SOURCES:%.cc=%.o)

CXXFLAGS += $(OMP_FLAGS)

#Libboost
LDFLAGS += -lboost_program_options -lconfig++

all: objs singen

singen: $(OBJECTS)
	$(CC) $(CXXFLAGS) $(DEFS) -o $@ $(OBJECTS) $(LDFLAGS)
	
objs: $(SOURCES)
	$(CC) $(CXXFLAGS) $(DEFS) -c $^ $< 

clean:
	- $(RM) *.o csr4gen singen

