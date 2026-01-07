#include "cublas_matmul.h"

#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>

using namespace cublas_test;

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nOptions:\n"
              << "  -m <size>      Number of rows in A and C (default: 1024)\n"
              << "  -n <size>      Number of cols in B and C (default: 1024)\n"
              << "  -k <size>      Shared dimension (cols of A, rows of B) (default: 1024)\n"
              << "  -s <size>      Set M=N=K to this value (square matrices)\n"
              << "  -i <mode>      Initialization mode for A and B:\n"
              << "                   zeros, 0   - All zeros\n"
              << "                   ones, 1    - All ones\n"
              << "                   random, r  - Random values in [0,1)\n"
              << "                   identity, i - Identity matrix\n"
              << "                   <number>   - Custom value (e.g., 2.5)\n"
              << "  -v             Verify result against CPU computation\n"
              << "  -p             Print matrices (only for small matrices)\n"
              << "  -r <count>     Number of repetitions for timing (default: 1)\n"
              << "  -w <count>     Number of warmup iterations (default: 1)\n"
              << "  -h             Show this help message\n"
              << "\nExamples:\n"
              << "  " << prog << " -s 2048 -i random -v     # 2048x2048 random matrices with verification\n"
              << "  " << prog << " -m 512 -n 1024 -k 768    # Non-square matrices\n"
              << "  " << prog << " -s 4096 -i ones -r 10    # Benchmark with 10 repetitions\n"
              << "  " << prog << " -s 64 -i 1 -p -v         # Small matrix test with printing\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;
    std::string init_str = "random";
    bool verify = false;
    bool print_matrices = false;
    int repetitions = 1;
    int warmup = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-m" && i + 1 < argc) {
            M = std::stoull(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            N = std::stoull(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            K = std::stoull(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            M = N = K = std::stoull(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            init_str = argv[++i];
        } else if (arg == "-v") {
            verify = true;
        } else if (arg == "-p") {
            print_matrices = true;
        } else if (arg == "-r" && i + 1 < argc) {
            repetitions = std::stoi(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Parse initialization mode
    InitMode init_mode = stringToInitMode(init_str);
    float custom_value = 0.0f;
    if (init_mode == InitMode::CUSTOM) {
        try {
            custom_value = std::stof(init_str);
        } catch (...) {
            std::cerr << "Invalid init mode or value: " << init_str << "\n";
            return 1;
        }
    }

    std::cout << "============================================\n";
    std::cout << "cuBLAS Matrix Multiplication Test\n";
    std::cout << "============================================\n";
    std::cout << "Matrix dimensions:\n";
    std::cout << "  A: " << M << " x " << K << "\n";
    std::cout << "  B: " << K << " x " << N << "\n";
    std::cout << "  C: " << M << " x " << N << "\n";
    std::cout << "Initialization: " << init_str << "\n";
    std::cout << "Repetitions: " << repetitions << " (warmup: " << warmup << ")\n";
    std::cout << "--------------------------------------------\n";

    try {
        // Create matrices
        std::cout << "Allocating matrices...\n";
        Matrix A(M, K);
        Matrix B(K, N);
        Matrix C(M, N);

        // Initialize matrices
        std::cout << "Initializing matrices with mode: " << init_str << "...\n";
        A.init(init_mode, custom_value);
        B.init(init_mode, custom_value);
        C.init(InitMode::ZEROS);

        // Print matrices if requested (only for small matrices)
        if (print_matrices) {
            A.print("Matrix A");
            B.print("Matrix B");
        }

        // Copy to device
        std::cout << "Copying matrices to GPU...\n";
        A.copyToDevice();
        B.copyToDevice();
        C.copyToDevice();

        // Create cuBLAS context
        CublasMatMul matmul;

        // Warmup runs
        std::cout << "Running " << warmup << " warmup iteration(s)...\n";
        for (int i = 0; i < warmup; ++i) {
            matmul.multiply(A, B, C);
        }

        // Timed runs
        std::cout << "Running " << repetitions << " timed iteration(s)...\n";
        float total_time = 0.0f;
        float total_gflops = 0.0f;
        float min_time = 1e9f;
        float max_time = 0.0f;

        for (int i = 0; i < repetitions; ++i) {
            // Reset C for each iteration if needed
            if (i > 0) {
                C.init(InitMode::ZEROS);
                C.copyToDevice();
            }
            
            matmul.multiply(A, B, C);
            
            float time_ms = matmul.getLastTimeMs();
            float gflops = matmul.getLastGflops();
            
            total_time += time_ms;
            total_gflops += gflops;
            min_time = std::min(min_time, time_ms);
            max_time = std::max(max_time, time_ms);
            
            if (repetitions <= 10 || i == 0 || i == repetitions - 1) {
                std::cout << "  Iteration " << (i + 1) << ": " 
                          << std::fixed << std::setprecision(3) << time_ms << " ms, "
                          << std::setprecision(2) << gflops << " GFLOPS\n";
            } else if (i == 1) {
                std::cout << "  ...\n";
            }
        }

        // Copy result back to host
        C.copyToHost();

        // Print result matrix if requested
        if (print_matrices) {
            C.print("Matrix C (Result)");
        }

        // Print statistics
        std::cout << "--------------------------------------------\n";
        std::cout << "Results:\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Average time: " << (total_time / repetitions) << " ms\n";
        std::cout << "  Min time:     " << min_time << " ms\n";
        std::cout << "  Max time:     " << max_time << " ms\n";
        std::cout << std::setprecision(2);
        std::cout << "  Average GFLOPS: " << (total_gflops / repetitions) << "\n";
        
        // Calculate memory bandwidth
        size_t bytes_read = (M * K + K * N) * sizeof(float);
        size_t bytes_write = M * N * sizeof(float);
        float bandwidth_gb = (bytes_read + bytes_write) / (min_time * 1e6f);
        std::cout << "  Memory bandwidth (peak): " << bandwidth_gb << " GB/s\n";

        // Verification
        if (verify) {
            std::cout << "--------------------------------------------\n";
            std::cout << "Verifying result against CPU computation...\n";
            if (!verifyResult(A, B, C)) {
                std::cerr << "VERIFICATION FAILED!\n";
                return 1;
            }
        }

        std::cout << "============================================\n";
        std::cout << "Test completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

