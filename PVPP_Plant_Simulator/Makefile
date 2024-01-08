
PROGRAM=$(PREFIX)executable
PREFIX=./bin/

.DEFAULT_GOAL:=$(PROGRAM)

PWD := $(shell pwd)

SRC_CPP = $(wildcard *.cpp)  
SRC_C = $(wildcard AmdLib/*.c) 
SRC_CUDA = $(wildcard *.cu) 	

DEPS = 	$(patsubst %.cpp,%.o,$(SRC_CPP)) $(patsubst %.c,%.o,$(SRC_C))  $(patsubst %.cu,%.o,$(SRC_CUDA))
DEPS_PREFIX = $(addprefix $(PREFIX),$(DEPS))

DEPENDS := $(addprefix $(PREFIX),  $(patsubst %.cpp,%.d,$(SRC_CPP)), $(patsubst %.c,%.d,$(SRC_C)),  $(patsubst %.cu,%.d,$(SRC_CUDA))  )
-include $(DEPENDS)

FLAGSEXTRA= -Wall -Wpedantic -pedantic  \
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

LINKFLAGS= -L /usr/local/cuda/lib64  -lcudart -lcuda -lpthread -lm  -lgomp -lgomp

OPTIMIZATION=-O3

INCLUDE_PATH= -I. -IAmdLib 
CFLAGS= $(OPTIMIZATION) $(INCLUDE_PATH) -g -march=native -mtune=native -pipe -fPIC  -funroll-loops -Wno-unknown-pragmas $(FLAGSEXTRA) $(DEFINES) --fast-math
NVCCDEBUG= 

NVCCFLAGS=$(NVCCDEBUG) $(OPTIMIZATION) $(INCLUDE_PATH) -arch=sm_86 -w -m64 -Xptxas -dlcm=cg -ftz=true --ptxas-options=-v --compiler-options "-D_FORCE_INLINES -fno-strict-aliasing -Wall $(OPTIMIZATION) $(DEFINES)" -DUNIX

.PHONY: clean

all: $(PROGRAM)

$(PROGRAM): $(DEPS_PREFIX)
	@echo "PVFARM" $(PROGRAM)
	@mkdir -p $(PREFIX)
	@/usr/bin/g++ $(CXXFLAGS)  $^ $(LINKFLAGS) -o $@

$(PREFIX)%.o: %.cu Makefile
	@echo "CUDA " $<
	@mkdir -p $(dir $@)
	@nvcc -std=c++17 $(NVCCFLAGS)  -MD -I/usr/local/cuda/include   -o $@ -c $<

$(PREFIX)%.o: %.cpp Makefile
	@echo "G++  " $<
	@mkdir -p $(dir $@)
	@g++ -std=c++20 -fopenmp $(CFLAGS) -MMD -MP -o $@ -c $<

$(PREFIX)%.o: %.c Makefile
	@echo "GCC  " $<
	@mkdir -p $(dir $@)
	@gcc -fopenmp $(CFLAGS) -MMD -MP -o $@ -c $<
	
clean:
	rm -rf $(PREFIX)

run: all
#$(PROGRAM)
	$(PROGRAM)
	




