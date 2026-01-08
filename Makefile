# Makefile for cuBLAS Matrix Multiplication Testing

# CUDA paths - adjust if needed
CUDA_PATH ?= /usr/local/cuda
CUDA_INC  := $(CUDA_PATH)/include
CUDA_LIB  := $(CUDA_PATH)/lib64

# Compilers
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

# GPU_ARCH  := -gencode arch=compute_120,code=sm_120
GPU_ARCH  := -gencode arch=compute_70,code=sm_70

# Compiler flags
NVCC_FLAGS := -std=c++17 $(GPU_ARCH) -O3 -Xcompiler -Wall,-Wextra
CXX_FLAGS  := -std=c++17 -O3 -Wall -Wextra -I$(CUDA_INC)
LDFLAGS    := -L$(CUDA_LIB) -lcudart -lcublas

# Debug build flags (use: make DEBUG=1)
ifdef DEBUG
    NVCC_FLAGS += -g -G -DDEBUG
    CXX_FLAGS  += -g -DDEBUG
endif

# Source files
CUDA_SOURCES := cublas_matmul.cu
CXX_SOURCES  := basicPerf.cc

# Object files
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
CXX_OBJECTS  := $(CXX_SOURCES:.cc=.o)

# Targets
TARGETS := basicPerf singleThread multiThread

# Default target
.PHONY: all
all: $(TARGETS)

# Build basicPerf executable
basicPerf: basicPerf.o cublas_matmul.o
	$(NVCC) $(GPU_ARCH) -o $@ $^ $(LDFLAGS)

# Build singleThread executable (warmup analysis)
singleThread: singleThread.o cublas_matmul.o
	$(NVCC) $(GPU_ARCH) -o $@ $^ $(LDFLAGS)

# Build multiThread executable (multi-threaded test)
multiThread: multiThread.o cublas_matmul.o
	$(NVCC) $(GPU_ARCH) -o $@ $^ $(LDFLAGS) -lpthread

# Compile CUDA source files
%.o: %.cu cublas_matmul.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Compile C++ source files with nvcc (for CUDA runtime support)
%.o: %.cc cublas_matmul.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(TARGETS) *.o

# Run tests
.PHONY: run
run: basicPerf
	./basicPerf

# Run singleThread (warmup analysis)
.PHONY: run2
run2: singleThread
	./singleThread

# Run multiThread (multi-threaded test)
.PHONY: run3
run3: multiThread
	./multiThread

# Run singleThread with larger matrices for more visible warmup effects
.PHONY: warmup-analysis
warmup-analysis: singleThread
	@echo "Running warmup analysis on NVIDIA GPU..."
	@echo ""
	@echo "=== 2048x2048 (5 warmup, 15 timed) ==="
	./singleThread -s 2048 -w 5 -r 15
	@echo ""
	@echo "=== 4096x4096 (5 warmup, 15 timed) ==="
	./singleThread -s 4096 -w 5 -r 15

# Run a quick test with small matrices
.PHONY: test-small
test-small: basicPerf
	./basicPerf -s 64 -i ones -p -v

# Run benchmark with various sizes
.PHONY: benchmark
benchmark: basicPerf
	@echo "Running benchmarks on NVIDIA GPU..."
	@echo ""
	@echo "=== 1024x1024 ==="
	./basicPerf -s 1024 -i random -r 10 -w 3
	@echo ""
	@echo "=== 2048x2048 ==="
	./basicPerf -s 2048 -i random -r 10 -w 3
	@echo ""
	@echo "=== 4096x4096 ==="
	./basicPerf -s 4096 -i random -r 10 -w 3
	@echo ""
	@echo "=== 8192x8192 ==="
	./basicPerf -s 8192 -i random -r 10 -w 3

# Help
.PHONY: help
help:
	@echo "cuBLAS Matrix Multiplication Test - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all             - Build all executables (default)"
	@echo "  basicPerf       - Build the main test executable"
	@echo "  singleThread    - Build the warmup analysis executable"
	@echo "  multiThread     - Build the multi-threaded test executable"
	@echo "  clean           - Remove build artifacts"
	@echo "  run             - Build and run basicPerf with default options"
	@echo "  run2            - Build and run singleThread with default options"
	@echo "  run3            - Build and run multiThread with default options"
	@echo "  warmup-analysis - Run detailed warmup analysis with singleThread"
	@echo "  test-small      - Run a quick test with small matrices"
	@echo "  benchmark       - Run benchmarks with various matrix sizes"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1         - Build with debug symbols"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build everything"
	@echo "  make singleThread       # Build singleThread (warmup analysis)"
	@echo "  make multiThread        # Build multiThread (multi-threaded test)"
	@echo "  make run2               # Build and run singleThread"
	@echo "  make run3               # Build and run multiThread"
	@echo "  make warmup-analysis    # Run detailed warmup analysis"
	@echo "  make DEBUG=1 basicPerf  # Build with debug symbols"
	@echo "  make benchmark          # Run benchmarks"

