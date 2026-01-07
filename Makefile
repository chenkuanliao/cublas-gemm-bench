# Makefile for cuBLAS Matrix Multiplication Testing
# Optimized for NVIDIA V100 GPU

# CUDA paths - adjust if needed
CUDA_PATH ?= /usr/local/cuda
CUDA_INC  := $(CUDA_PATH)/include
CUDA_LIB  := $(CUDA_PATH)/lib64

# Compilers
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

# V100 has compute capability 7.0
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
CXX_SOURCES  := test1.cc

# Object files
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
CXX_OBJECTS  := $(CXX_SOURCES:.cc=.o)

# Targets
TARGETS := test1 test2

# Default target
.PHONY: all
all: $(TARGETS)

# Build test1 executable
test1: test1.o cublas_matmul.o
	$(NVCC) $(GPU_ARCH) -o $@ $^ $(LDFLAGS)

# Build test2 executable (warmup analysis)
test2: test2.o cublas_matmul.o
	$(NVCC) $(GPU_ARCH) -o $@ $^ $(LDFLAGS)

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
run: test1
	./test1

# Run test2 (warmup analysis)
.PHONY: run2
run2: test2
	./test2

# Run test2 with larger matrices for more visible warmup effects
.PHONY: warmup-analysis
warmup-analysis: test2
	@echo "Running warmup analysis on V100..."
	@echo ""
	@echo "=== 2048x2048 (5 warmup, 15 timed) ==="
	./test2 -s 2048 -w 5 -r 15
	@echo ""
	@echo "=== 4096x4096 (5 warmup, 15 timed) ==="
	./test2 -s 4096 -w 5 -r 15

# Run a quick test with small matrices
.PHONY: test-small
test-small: test1
	./test1 -s 64 -i ones -p -v

# Run benchmark with various sizes
.PHONY: benchmark
benchmark: test1
	@echo "Running benchmarks on V100..."
	@echo ""
	@echo "=== 1024x1024 ==="
	./test1 -s 1024 -i random -r 10 -w 3
	@echo ""
	@echo "=== 2048x2048 ==="
	./test1 -s 2048 -i random -r 10 -w 3
	@echo ""
	@echo "=== 4096x4096 ==="
	./test1 -s 4096 -i random -r 10 -w 3
	@echo ""
	@echo "=== 8192x8192 ==="
	./test1 -s 8192 -i random -r 10 -w 3

# Help
.PHONY: help
help:
	@echo "cuBLAS Matrix Multiplication Test - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all             - Build all executables (default)"
	@echo "  test1           - Build the main test executable"
	@echo "  test2           - Build the warmup analysis executable"
	@echo "  clean           - Remove build artifacts"
	@echo "  run             - Build and run test1 with default options"
	@echo "  run2            - Build and run test2 with default options"
	@echo "  warmup-analysis - Run detailed warmup analysis with test2"
	@echo "  test-small      - Run a quick test with small matrices"
	@echo "  benchmark       - Run benchmarks with various matrix sizes"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1         - Build with debug symbols"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build everything"
	@echo "  make test2              # Build test2 (warmup analysis)"
	@echo "  make run2               # Build and run test2"
	@echo "  make warmup-analysis    # Run detailed warmup analysis"
	@echo "  make DEBUG=1 test1      # Build with debug symbols"
	@echo "  make benchmark          # Run benchmarks"

