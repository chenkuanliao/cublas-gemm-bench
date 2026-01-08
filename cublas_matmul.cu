#include "cublas_matmul.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <cstring>

namespace cublas_test {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(err) +                  \
                                     " at " + __FILE__ + ":" +                  \
                                     std::to_string(__LINE__));                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            throw std::runtime_error(std::string("cuBLAS error: ") +            \
                                     std::to_string(status) +                   \
                                     " at " + __FILE__ + ":" +                  \
                                     std::to_string(__LINE__));                 \
        }                                                                       \
    } while (0)

// ============================================================================
// Matrix Implementation
// ============================================================================

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), h_data_(nullptr), d_data_(nullptr) {
    
    size_t bytes = rows_ * cols_ * sizeof(float);
    
    // Allocate host memory (pinned for faster transfers)
    CUDA_CHECK(cudaMallocHost(&h_data_, bytes));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_data_, bytes));
    
    // Initialize to zero
    std::memset(h_data_, 0, bytes);
    CUDA_CHECK(cudaMemset(d_data_, 0, bytes));
}

Matrix::~Matrix() {
    if (h_data_) {
        cudaFreeHost(h_data_);
        h_data_ = nullptr;
    }
    if (d_data_) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_),
      h_data_(other.h_data_), d_data_(other.d_data_) {
    other.h_data_ = nullptr;
    other.d_data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (h_data_) cudaFreeHost(h_data_);
        if (d_data_) cudaFree(d_data_);
        
        // Move from other
        rows_ = other.rows_;
        cols_ = other.cols_;
        h_data_ = other.h_data_;
        d_data_ = other.d_data_;
        
        other.h_data_ = nullptr;
        other.d_data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

void Matrix::init(InitMode mode, float custom_value) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            size_t idx = i * cols_ + j;
            switch (mode) {
                case InitMode::ZEROS:
                    h_data_[idx] = 0.0f;
                    break;
                case InitMode::ONES:
                    h_data_[idx] = 1.0f;
                    break;
                case InitMode::RANDOM:
                    h_data_[idx] = dist(gen);
                    break;
                case InitMode::IDENTITY:
                    h_data_[idx] = (i == j) ? 1.0f : 0.0f;
                    break;
                case InitMode::CUSTOM:
                    h_data_[idx] = custom_value;
                    break;
            }
        }
    }
}

void Matrix::copyToDevice() {
    CUDA_CHECK(cudaMemcpy(d_data_, h_data_, 
                          rows_ * cols_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Matrix::copyToHost() {
    CUDA_CHECK(cudaMemcpy(h_data_, d_data_, 
                          rows_ * cols_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void Matrix::print(const std::string& name, size_t max_rows, size_t max_cols) const {
    if (!name.empty()) {
        std::cout << name << " (" << rows_ << "x" << cols_ << "):\n";
    }
    
    size_t print_rows = std::min(rows_, max_rows);
    size_t print_cols = std::min(cols_, max_cols);
    
    for (size_t i = 0; i < print_rows; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < print_cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << h_data_[i * cols_ + j];
            if (j < print_cols - 1) std::cout << ", ";
        }
        if (cols_ > max_cols) std::cout << ", ...";
        std::cout << "]\n";
    }
    if (rows_ > max_rows) {
        std::cout << "  ...\n";
    }
}

// ============================================================================
// CublasMatMul Implementation
// ============================================================================

CublasMatMul::CublasMatMul()
    : cublas_handle_(nullptr),
      stream_(0),
      owns_stream_(false),
      last_time_ms_(0),
      last_gflops_(0) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublas_handle_ = handle;
    
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
}

CublasMatMul::~CublasMatMul() {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
    if (owns_stream_ && stream_) {
        cudaStreamDestroy(stream_);
        stream_ = 0;
        owns_stream_ = false;
    }
}

void CublasMatMul::setStream(cudaStream_t stream, bool take_ownership) {
    if (owns_stream_ && stream_ && stream_ != stream) {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    stream_ = stream;
    owns_stream_ = take_ownership;

    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle_);
    CUBLAS_CHECK(cublasSetStream(handle, stream_));
}

void CublasMatMul::multiply(const Matrix& A, const Matrix& B, Matrix& C,
                            float alpha, float beta) {
    // Validate dimensions
    // A is (M x K), B is (K x N), C is (M x N)
    size_t M = A.rows();
    size_t K = A.cols();
    size_t N = B.cols();
    
    if (B.rows() != K) {
        throw std::runtime_error("Matrix dimension mismatch: A.cols (" + 
                                 std::to_string(K) + ") != B.rows (" + 
                                 std::to_string(B.rows()) + ")");
    }
    if (C.rows() != M || C.cols() != N) {
        throw std::runtime_error("Output matrix C has wrong dimensions");
    }
    
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle_);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // cuBLAS uses column-major order, but our matrices are row-major
    // To compute C = A * B in row-major, we compute C^T = B^T * A^T in column-major
    // This is equivalent to calling cublasSgemm with swapped A and B
    
    CUDA_CHECK(cudaEventRecord(start, stream_));
    
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N,    // B not transposed
                             CUBLAS_OP_N,    // A not transposed
                             N,              // Number of rows of B^T (= cols of B)
                             M,              // Number of cols of A^T (= rows of A)
                             K,              // Shared dimension
                             &alpha,
                             B.deviceData(), // B in row-major = B^T in column-major
                             N,              // Leading dimension of B
                             A.deviceData(), // A in row-major = A^T in column-major
                             K,              // Leading dimension of A
                             &beta,
                             C.deviceData(), // C in row-major
                             N));            // Leading dimension of C
    
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&last_time_ms_, start, stop));
    
    // Calculate GFLOPS: 2*M*N*K floating point operations
    double flops = 2.0 * M * N * K;
    last_gflops_ = static_cast<float>(flops / (last_time_ms_ * 1e6));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string initModeToString(InitMode mode) {
    switch (mode) {
        case InitMode::ZEROS:    return "zeros";
        case InitMode::ONES:     return "ones";
        case InitMode::RANDOM:   return "random";
        case InitMode::IDENTITY: return "identity";
        case InitMode::CUSTOM:   return "custom";
        default:                 return "unknown";
    }
}

InitMode stringToInitMode(const std::string& str) {
    if (str == "zeros" || str == "0")   return InitMode::ZEROS;
    if (str == "ones" || str == "1")    return InitMode::ONES;
    if (str == "random" || str == "r")  return InitMode::RANDOM;
    if (str == "identity" || str == "i") return InitMode::IDENTITY;
    return InitMode::CUSTOM;
}

bool verifyResult(const Matrix& A, const Matrix& B, const Matrix& C,
                  float alpha, float beta, float tolerance) {
    size_t M = A.rows();
    size_t K = A.cols();
    size_t N = B.cols();
    
    // Compute reference on CPU
    std::vector<float> ref(M * N, 0.0f);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A.hostData()[i * K + k] * B.hostData()[k * N + j];
            }
            ref[i * N + j] = alpha * sum;  // beta * C_old is not tracked
        }
    }
    
    // Compare with GPU result
    float max_diff = 0.0f;
    size_t error_count = 0;
    
    for (size_t i = 0; i < M * N; ++i) {
        float diff = std::abs(ref[i] - C.hostData()[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > tolerance) {
            error_count++;
            if (error_count <= 5) {
                std::cerr << "Mismatch at index " << i 
                          << ": expected " << ref[i] 
                          << ", got " << C.hostData()[i] 
                          << " (diff: " << diff << ")\n";
            }
        }
    }
    
    if (error_count > 0) {
        std::cerr << "Total errors: " << error_count << " / " << M * N 
                  << ", max diff: " << max_diff << "\n";
        return false;
    }
    
    std::cout << "Verification PASSED (max diff: " << max_diff << ")\n";
    return true;
}

} // namespace cublas_test

