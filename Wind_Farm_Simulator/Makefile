PROGRAM=$(PREFIX)executable
PREFIX=./bin/


.DEFAULT_GOAL:=$(PROGRAM)

SRC_CPP = $(wildcard *.cpp)

SRC_CUDA = $(wildcard *.cu)

DEPS = 		bin/build_flags.o  \
			$(patsubst %.cpp,%.o,$(SRC_CPP)) \
			$(patsubst %.cu,%.o,$(SRC_CUDA))

PREFIX_DEPS = $(addprefix $(PREFIX),$(DEPS))

DEPENDS := $(addprefix $(PREFIX),  $(patsubst %.cpp,%.d,$(SRC_CPP)) $(patsubst %.cu,%.d,$(SRC_CUDA))  )
-include $(DEPENDS)

EXTRA_FLAGS= -Wall -Wpedantic -pedantic  \
-Wextra -Wcast-align \
-Wcast-qual -Wconversion \
-Wdisabled-optimization \
-Wfloat-equal -Wformat=2 \
-Wformat-nonliteral -Wformat-security  \
-Wformat-y2k \
-Wimport  -Winit-self  -Winline \
-Winvalid-pch   \
-Wmissing-field-initializers -Wmissing-format-attribute   \
-Wmissing-include-dirs \
-Wpacked -Wpointer-arith \
-Wredundant-decls \
-Wshadow -Wstack-protector \
-Wstrict-aliasing=2 -Wswitch-default \
-Wswitch-enum \
-Wunreachable-code -Wunused \
-Wunused-parameter \
-Wvariadic-macros \
-Wwrite-strings  -Werror


OPENMP_FLAG=-fopenmp
OPTIMIZATION=-O3
INCLUDE_PATH= -I. 
CXXFLAGS= $(OPTIMIZATION) $(INCLUDE_PATH) -march=native -mtune=native -pipe -fPIC  -funroll-loops -Wno-unknown-pragmas $(EXTRA_FLAGS) $(DEFINES) --fast-math -std=c++11  $(OPENMP_FLAG)
NVCCDEBUG= 
NVCCFLAGS=$(NVCCDEBUG) $(OPTIMIZATION) $(INCLUDE_PATH) -arch=sm_86  -ftz=true --ptxas-options=-v --compiler-options "-D_FORCE_INLINES -fno-strict-aliasing -Wall $(OPTIMIZATION) $(DEFINES) $(OPENMP_FLAG)" -DUNIX

LINKFLAGS= -L /usr/local/cuda/lib64 -lcudart -lpthread -lm  -lblas
DEFINES=   # -DUSE_DOUBLE

.PHONY: clean

all: $(PROGRAM)

$(PROGRAM): $(PREFIX_DEPS)
	echo "WT_SIMULATOR" $(PROGRAM)
	@mkdir -p $(PREFIX)
	@/usr/bin/g++ $(CXXFLAGS)  $^ $(LINKFLAGS) -o $@

bin/build_flags.cpp: Makefile
	@echo "Build Flags"
	@mkdir -p bin
	@ if [ -s $@ ] ; then  truncate -s 0  $@; fi  
	@echo "#include <stdio.h>" >> $@
	@echo "void print_build_flags(){" >> $@
	@echo "puts(\" CXXFLAGS:  $(CXXFLAGS)\");"    >> $@
	@echo "puts(\" NVCCFLAGS: $(NVCCFLAGS)\");"   >> $@
	@echo "puts(\" LINKFLAGS: $(LINKFLAGS)\");"   >> $@
	@echo "}"                                     >> $@


$(PREFIX)%.o: %.cu Makefile
	@echo "CUDA " $<
	@mkdir -p $(dir $@)
	@nvcc -std=c++14 $(NVCCFLAGS)  -MD -I/usr/local/cuda/include   -o $@ -c $<

$(PREFIX)%.o: %.cpp Makefile
	@echo "G++  " $<
	@mkdir -p $(dir $@)
	@g++ -std=c++14 $(CXXFLAGS) -MMD -MP -o $@ -c $<

clean:
	rm -rf $(PREFIX)

run: all
	#paste -d"," inputs/Voltages_VoltageDip.csv inputs/Wind_Static.csv > bin/Input.csv
	#paste -d"," inputs/Voltages_Static.csv inputs/Wind_Static.csv > bin/Input.csv

	paste -d"," inputs/Voltages_VoltageDip.csv inputs/Wind_Gust.csv > bin/Input.csv
	
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 ;  $(PROGRAM)
	
