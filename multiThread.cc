#include "cublas_matmul.h"

#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

using namespace cublas_test;

std::mutex print_mutex;

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
              << "  -w <count>     Number of warmup iterations on parent thread (default: 3)\n"
              << "  -r <count>     Number of timed iterations per child thread (default: 10)\n"
              << "  -t <count>     Number of child threads (default: 4)\n"
              << "  -h             Show this help message\n"
              << "\nThis test does warmups on the parent thread, then spawns child threads\n"
              << "where each child thread performs cuBLAS matrix multiplications.\n";
}

void printSeparator(char c = '-', int width = 70) {
    std::cout << std::string(width, c) << "\n";
}

struct ThreadResult {
    int thread_id;
    std::vector<float> times_ms;
    std::vector<float> gflops;
    float total_time_ms;
};

void threadWorker(int thread_id, 
                  size_t M, size_t N, size_t K,
                  InitMode init_mode, float custom_value,
                  int iterations,
                  ThreadResult& result) {
    try {
        // Each thread creates its own matrices and cuBLAS handle
        Matrix A(M, K);
        Matrix B(K, N);
        Matrix C(M, N);

        A.init(init_mode, custom_value);
        B.init(init_mode, custom_value);
        C.init(InitMode::ZEROS);

        A.copyToDevice();
        B.copyToDevice();
        C.copyToDevice();

        CublasMatMul matmul;

        result.thread_id = thread_id;
        result.times_ms.reserve(iterations);
        result.gflops.reserve(iterations);

        auto thread_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            C.init(InitMode::ZEROS);
            C.copyToDevice();

            matmul.multiply(A, B, C);

            float time_ms = matmul.getLastTimeMs();
            float gflops_val = matmul.getLastGflops();
            result.times_ms.push_back(time_ms);
            result.gflops.push_back(gflops_val);
        }

        auto thread_end = std::chrono::high_resolution_clock::now();
        result.total_time_ms = std::chrono::duration<float, std::milli>(thread_end - thread_start).count();

        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "  Thread " << std::setw(2) << thread_id << " completed: "
                      << iterations << " iterations in "
                      << std::fixed << std::setprecision(2) << result.total_time_ms << " ms\n";
        }

    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "  Thread " << thread_id << " error: " << e.what() << "\n";
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;
    std::string init_str = "random";
    int warmup_count = 1;
    int timed_count = 10;
    int num_threads = 4;

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
        } else if (arg == "-t" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
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
    std::cout << "  cuBLAS Multi-Threaded Performance Test\n";
    printSeparator('=');
    std::cout << "Matrix dimensions: " << M << " x " << K << " * " << K << " x " << N << "\n";
    std::cout << "Warmup iterations (parent thread): " << warmup_count << "\n";
    std::cout << "Timed iterations (per child thread): " << timed_count << "\n";
    std::cout << "Number of child threads: " << num_threads << "\n";
    printSeparator();

    try {
        // ============================================
        // WARMUP PHASE ON PARENT THREAD
        // ============================================
        printSeparator();
        std::cout << "  WARMUP PHASE (Parent Thread, " << warmup_count << " iterations)\n";
        printSeparator();

        // Create matrices for warmup
        std::cout << "Allocating and initializing warmup matrices...\n";
        Matrix warmup_A(M, K);
        Matrix warmup_B(K, N);
        Matrix warmup_C(M, N);

        warmup_A.init(init_mode, custom_value);
        warmup_B.init(init_mode, custom_value);
        warmup_C.init(InitMode::ZEROS);

        warmup_A.copyToDevice();
        warmup_B.copyToDevice();
        warmup_C.copyToDevice();

        CublasMatMul warmup_matmul;

        std::vector<float> warmup_times;
        warmup_times.reserve(warmup_count);

        for (int i = 0; i < warmup_count; ++i) {
            warmup_C.init(InitMode::ZEROS);
            warmup_C.copyToDevice();

            warmup_matmul.multiply(warmup_A, warmup_B, warmup_C);

            float time_ms = warmup_matmul.getLastTimeMs();
            float gflops = warmup_matmul.getLastGflops();
            warmup_times.push_back(time_ms);

            std::cout << "  Warmup " << std::setw(2) << (i + 1) << ": "
                      << std::fixed << std::setprecision(3) << std::setw(10) << time_ms << " ms  "
                      << std::setprecision(2) << std::setw(10) << gflops << " GFLOPS\n";
        }

        float warmup_avg = std::accumulate(warmup_times.begin(), warmup_times.end(), 0.0f) / warmup_times.size();
        std::cout << "  Warmup average: " << std::fixed << std::setprecision(3) << warmup_avg << " ms\n";

        // ============================================
        // CHILD THREAD PHASE
        // ============================================
        printSeparator();
        std::cout << "  CHILD THREAD PHASE (" << num_threads << " threads, " 
                  << timed_count << " iterations each)\n";
        printSeparator();

        std::vector<std::thread> threads;
        std::vector<ThreadResult> results(num_threads);

        auto overall_start = std::chrono::high_resolution_clock::now();

        // Spawn child threads
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(threadWorker, t, M, N, K, 
                                 init_mode, custom_value, 
                                 timed_count, std::ref(results[t]));
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        auto overall_end = std::chrono::high_resolution_clock::now();
        float overall_time_ms = std::chrono::duration<float, std::milli>(overall_end - overall_start).count();

        // ============================================
        // PER-THREAD STATISTICS
        // ============================================
        printSeparator();
        std::cout << "  PER-THREAD STATISTICS\n";
        printSeparator();

        double flops = 2.0 * M * N * K;
        auto ms_to_gflops = [flops](float ms) { return (flops / (ms * 1e6)); };

        for (int t = 0; t < num_threads; ++t) {
            const auto& r = results[t];
            if (r.times_ms.empty()) continue;

            float avg_time = std::accumulate(r.times_ms.begin(), r.times_ms.end(), 0.0f) / r.times_ms.size();
            float min_time = *std::min_element(r.times_ms.begin(), r.times_ms.end());
            float max_time = *std::max_element(r.times_ms.begin(), r.times_ms.end());

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Thread " << std::setw(2) << t << ":\n";
            
            // Print individual run timings
            for (size_t i = 0; i < r.times_ms.size(); ++i) {
                std::cout << "    Run " << std::setw(3) << (i + 1) << ": "
                          << std::setw(10) << r.times_ms[i] << " ms  ("
                          << std::setprecision(2) << std::setw(8) << r.gflops[i] << " GFLOPS)\n";
                std::cout << std::setprecision(3);
            }
            
            std::cout << "    --------\n";
            std::cout << "    Avg: " << std::setw(10) << avg_time << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(avg_time) << " GFLOPS)\n";
            std::cout << std::setprecision(3);
            std::cout << "    Min: " << std::setw(10) << min_time << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(min_time) << " GFLOPS)\n";
            std::cout << std::setprecision(3);
            std::cout << "    Max: " << std::setw(10) << max_time << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(max_time) << " GFLOPS)\n";
        }

        // ============================================
        // AGGREGATE STATISTICS
        // ============================================
        printSeparator();
        std::cout << "  AGGREGATE STATISTICS\n";
        printSeparator();

        // Collect all times across all threads
        std::vector<float> all_times;
        for (const auto& r : results) {
            all_times.insert(all_times.end(), r.times_ms.begin(), r.times_ms.end());
        }

        if (!all_times.empty()) {
            float agg_avg = std::accumulate(all_times.begin(), all_times.end(), 0.0f) / all_times.size();
            float agg_min = *std::min_element(all_times.begin(), all_times.end());
            float agg_max = *std::max_element(all_times.begin(), all_times.end());

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Total operations: " << all_times.size() << "\n";
            std::cout << "  Overall wall time: " << std::setw(10) << overall_time_ms << " ms\n";
            std::cout << "\n  All thread operations combined:\n";
            std::cout << "    Avg per-op: " << std::setw(10) << agg_avg << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(agg_avg) << " GFLOPS)\n";
            std::cout << std::setprecision(3);
            std::cout << "    Min per-op: " << std::setw(10) << agg_min << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(agg_min) << " GFLOPS)\n";
            std::cout << std::setprecision(3);
            std::cout << "    Max per-op: " << std::setw(10) << agg_max << " ms  ("
                      << std::setprecision(2) << std::setw(8) << ms_to_gflops(agg_max) << " GFLOPS)\n";

            // Aggregate throughput
            double total_gflops = (flops * all_times.size()) / (overall_time_ms * 1e6);
            
            std::cout << std::setprecision(2);
            std::cout << "\n  Aggregate throughput: " << total_gflops << " GFLOPS "
                      << "(total ops across " << num_threads << " threads / wall time)\n";
        }

        // ============================================
        // COMPARISON WITH SINGLE-THREADED WARMUP
        // ============================================
        printSeparator();
        std::cout << "  COMPARISON\n";
        printSeparator();

        if (!all_times.empty()) {
            float agg_avg = std::accumulate(all_times.begin(), all_times.end(), 0.0f) / all_times.size();
            
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Warmup avg (parent):     " << std::setw(10) << warmup_avg << " ms\n";
            std::cout << "  Child threads avg:       " << std::setw(10) << agg_avg << " ms\n";
            std::cout << std::setprecision(2);
            std::cout << "  Ratio (child/warmup):    " << std::setw(10) << (agg_avg / warmup_avg) << "x\n";
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

