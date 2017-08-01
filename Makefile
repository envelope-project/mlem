CC      = mpicxx
RM      = rm -f

-include Makefile.config

CFLAGS  = -Wall -std=c++11 -O3 -fstrict-aliasing
LFLAGS  = -Wall -O3
#CFLAGS  = -Wall -std=c++11 -g -fstrict-aliasing
#LFLAGS  = -Wall -g

SOURCES = csr4matrix.cpp scannerconfig.cpp
HEADERS = csr4matrix.hpp vector.hpp matrixelement.hpp scannerconfig.hpp
OBJECTS = $(SOURCES:%.cpp=%.o)

# LAIK lib
LAIKLIB = -L$(LAIK_ROOT)/ -llaik
CFLAGS += -I$(LAIK_ROOT)/include $(BOOST_INC)

all: mpicsr4mlem laikcsr4mlem laikcsr4mlem-repart

mpicsr4mlem: mpicsr4mlem.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ mpicsr4mlem.o $(OBJECTS)

laikcsr4mlem: laikcsr4mlem.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ laikcsr4mlem.o $(OBJECTS) $(LAIKLIB)

laikcsr4mlem-repart: laikcsr4mlem-repart.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ laikcsr4mlem-repart.o $(OBJECTS) $(LAIKLIB)

%.o: %.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $< 

clean:
	- $(RM) *.o mpicsr4mlem laikcsr4mlem

distclean: clean
	- $(RM) *.c~ *.h~
	
