CC      = g++
RM      = rm -f

#-include Makefile.config

CXXFLAGS  = -Wall -std=c++11 -O3 -fstrict-aliasing
LDFLAGS = 

SOURCES = singen.cc

all: singen

singen: $(OBJECTS)
	$(CC) $(CXXFLAGS) $(DEFS) -o $@ $(SOURCES) $(LDFLAGS)

clean:
	- $(RM) *.o singen

