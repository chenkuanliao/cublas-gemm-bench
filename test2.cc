#include "cublas_matmul.h"

#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

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
              << "  -w <count>     Number of warmup iterations (default: 3)\n"
              << "  -r <count>     Number of timed iterations (default: 10)\n"
              << "  -h             Show this help message\n"
              << "\nThis test shows timing breakdown for warmup, first run, and later runs.\n";
}

void printSeparator(char c = '-', int width = 60) {
    std::cout << std::string(width, c) << "\n";
}

void printTimingBar(float time_ms, float max_time_ms, int bar_width = 30) {
    int filled = static_cast<int>((time_ms / max_time_ms) * bar_width);
    filled = std::min(filled, bar_width);
    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        std::cout << (i < filled ? "█" : "░");
    }
    std::cout << "]";
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;
    std::string init_str = "random";
    int warmup_count = 1;
    int timed_count = 10;

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
        } else if (arg == "-w" && i + 1 < argc) {
            warmup_count = std::stoi(argv[++i]);
        } else if (arg == "-r" && i + 1 < argc) {
            timed_count = std::stoi(argv[++i]);
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

    std::cout << "\n";
    printSeparator('=');
    std::cout << "  cuBLAS Warmup & Performance Analysis\n";
    printSeparator('=');
    std::cout << "Matrix dimensions: " << M << " x " << K << " * " << K << " x " << N << "\n";
    std::cout << "Warmup iterations: " << warmup_count << "\n";
    std::cout << "Timed iterations:  " << timed_count << "\n";
    printSeparator();

    try {
        // Create matrices
        std::cout << "Allocating and initializing matrices...\n";
        Matrix A(M, K);
        Matrix B(K, N);
        Matrix C(M, N);

        A.init(init_mode, custom_value);
        B.init(init_mode, custom_value);
        C.init(InitMode::ZEROS);

        // Copy to device
        std::cout << "Copying matrices to GPU...\n";
        A.copyToDevice();
        B.copyToDevice();
        C.copyToDevice();

        // Create cuBLAS context
        CublasMatMul matmul;

        // Storage for all timings
        std::vector<float> warmup_times;
        std::vector<float> timed_times;

        // ============================================
        // WARMUP PHASE
        // ============================================
        printSeparator();
        std::cout << "  WARMUP PHASE (" << warmup_count << " iterations)\n";
        printSeparator();

        for (int i = 0; i < warmup_count; ++i) {
            C.init(InitMode::ZEROS);
            C.copyToDevice();
            
            matmul.multiply(A, B, C);
            
            float time_ms = matmul.getLastTimeMs();
            float gflops = matmul.getLastGflops();
            warmup_times.push_back(time_ms);
            
            std::cout << "  Warmup " << std::setw(2) << (i + 1) << ": "
                      << std::fixed << std::setprecision(3) << std::setw(10) << time_ms << " ms  "
                      << std::setprecision(2) << std::setw(10) << gflops << " GFLOPS\n";
        }

        // ============================================
        // FIRST TIMED RUN (after warmup)
        // ============================================
        printSeparator();
        std::cout << "  FIRST TIMED RUN (post-warmup)\n";
        printSeparator();

        C.init(InitMode::ZEROS);
        C.copyToDevice();
        matmul.multiply(A, B, C);
        
        float first_time = matmul.getLastTimeMs();
        float first_gflops = matmul.getLastGflops();
        timed_times.push_back(first_time);

        std::cout << "  First run: "
                  << std::fixed << std::setprecision(3) << std::setw(10) << first_time << " ms  "
                  << std::setprecision(2) << std::setw(10) << first_gflops << " GFLOPS\n";

        // ============================================
        // SUBSEQUENT RUNS
        // ============================================
        printSeparator();
        std::cout << "  SUBSEQUENT RUNS (" << (timed_count - 1) << " iterations)\n";
        printSeparator();

        for (int i = 1; i < timed_count; ++i) {
            C.init(InitMode::ZEROS);
            C.copyToDevice();
            
            matmul.multiply(A, B, C);
            
            float time_ms = matmul.getLastTimeMs();
            float gflops = matmul.getLastGflops();
            timed_times.push_back(time_ms);
            
            std::cout << "  Run " << std::setw(3) << (i + 1) << ": "
                      << std::fixed << std::setprecision(3) << std::setw(10) << time_ms << " ms  "
                      << std::setprecision(2) << std::setw(10) << gflops << " GFLOPS\n";
        }

        // ============================================
        // SUMMARY & COMPARISON
        // ============================================
        printSeparator('=');
        std::cout << "  PERFORMANCE SUMMARY\n";
        printSeparator('=');

        // Calculate statistics for warmup
        float warmup_first = warmup_times.front();
        float warmup_last = warmup_times.back();
        float warmup_avg = std::accumulate(warmup_times.begin(), warmup_times.end(), 0.0f) / warmup_times.size();
        float warmup_min = *std::min_element(warmup_times.begin(), warmup_times.end());
        float warmup_max = *std::max_element(warmup_times.begin(), warmup_times.end());

        // Calculate statistics for timed runs (excluding first)
        std::vector<float> later_times(timed_times.begin() + 1, timed_times.end());
        float later_avg = 0.0f, later_min = 0.0f, later_max = 0.0f;
        if (!later_times.empty()) {
            later_avg = std::accumulate(later_times.begin(), later_times.end(), 0.0f) / later_times.size();
            later_min = *std::min_element(later_times.begin(), later_times.end());
            later_max = *std::max_element(later_times.begin(), later_times.end());
        }

        // Calculate GFLOPS
        double flops = 2.0 * M * N * K;
        auto ms_to_gflops = [flops](float ms) { return (flops / (ms * 1e6)); };

        std::cout << std::fixed << std::setprecision(3);
        
        std::cout << "\n  Warmup Phase:\n";
        std::cout << "    First warmup:  " << std::setw(10) << warmup_first << " ms  ("
                  << std::setprecision(2) << std::setw(8) << ms_to_gflops(warmup_first) << " GFLOPS)\n";
        std::cout << std::setprecision(3);
        std::cout << "    Last warmup:   " << std::setw(10) << warmup_last << " ms  ("
                  << std::setprecision(2) << std::setw(8) << ms_to_gflops(warmup_last) << " GFLOPS)\n";
        std::cout << std::setprecision(3);
        std::cout << "    Avg warmup:    " << std::setw(10) << warmup_avg << " ms  ("
                  << std::setprecision(2) << std::setw(8) << ms_to_gflops(warmup_avg) << " GFLOPS)\n";

        std::cout << std::setprecision(3);
        std::cout << "\n  First Timed Run:\n";
        std::cout << "    Time:          " << std::setw(10) << first_time << " ms  ("
                  << std::setprecision(2) << std::setw(8) << ms_to_gflops(first_time) << " GFLOPS)\n";

        if (!later_times.empty()) {
            std::cout << std::setprecision(3);
            std::cout << "\n  Later Runs (runs 2-" << timed_count << "):\n";
            std::cout << "    Average:       " << std::setw(10) << later_avg << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(later_avg) << " GFLOPS)\n";
            std::cout << std::setprecision(3);
            std::cout << "    Min:           " << std::setw(10) << later_min << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(later_min) << " GFLOPS)\n";
            std::cout << std::setprecision(3);
            std::cout << "    Max:           " << std::setw(10) << later_max << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(later_max) << " GFLOPS)\n";
        }

        // ============================================
        // SPEEDUP ANALYSIS
        // ============================================
        printSeparator();
        std::cout << "  SPEEDUP ANALYSIS\n";
        printSeparator();

        float reference_time = (!later_times.empty()) ? later_avg : first_time;
        
        std::cout << std::setprecision(2);
        std::cout << "\n  Relative to steady-state (later runs avg):\n";
        std::cout << "    First warmup:   " << std::setw(6) << (warmup_first / reference_time) << "x slower";
        if (warmup_first > reference_time) {
            std::cout << "  (+" << std::setprecision(1) << ((warmup_first / reference_time - 1) * 100) << "% overhead)";
        }
        std::cout << "\n";
        
        std::cout << std::setprecision(2);
        std::cout << "    Last warmup:    " << std::setw(6) << (warmup_last / reference_time) << "x";
        if (warmup_last > reference_time * 1.05) {
            std::cout << " slower";
        } else if (warmup_last < reference_time * 0.95) {
            std::cout << " faster";
        } else {
            std::cout << " (similar)";
        }
        std::cout << "\n";

        std::cout << "    First timed:    " << std::setw(6) << (first_time / reference_time) << "x";
        if (first_time > reference_time * 1.05) {
            std::cout << " slower";
        } else if (first_time < reference_time * 0.95) {
            std::cout << " faster";
        } else {
            std::cout << " (similar)";
        }
        std::cout << "\n";

        // Visual comparison bar chart
        printSeparator();
        std::cout << "  VISUAL COMPARISON (time in ms)\n";
        printSeparator();
        
        float max_time = std::max({warmup_first, warmup_last, first_time, 
                                   (!later_times.empty() ? later_avg : first_time)});

        std::cout << std::setprecision(3);
        std::cout << "  1st warmup  " << std::setw(8) << warmup_first << " ms ";
        printTimingBar(warmup_first, max_time);
        std::cout << "\n";

        std::cout << "  Last warmup " << std::setw(8) << warmup_last << " ms ";
        printTimingBar(warmup_last, max_time);
        std::cout << "\n";

        std::cout << "  1st timed   " << std::setw(8) << first_time << " ms ";
        printTimingBar(first_time, max_time);
        std::cout << "\n";

        if (!later_times.empty()) {
            std::cout << "  Later avg   " << std::setw(8) << later_avg << " ms ";
            printTimingBar(later_avg, max_time);
            std::cout << "\n";
        }

        printSeparator('=');
        std::cout << "  Test completed successfully!\n";
        printSeparator('=');
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

