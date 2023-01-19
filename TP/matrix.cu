#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>


using namespace std;


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

__device__ void matrix_sum_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < m1->rows * m1->columns)
    { 
        res->m[index] = m1->m[index] + m2->m[index];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

__device__ void matrix_minus_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < m1->rows * m1->columns)
    { 
        res->m[index] = m1->m[index] - m2->m[index];
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}

void compareMatrix(double *A, double *B, int numRows, int numColumns){
    float sum = 0.0;
    float max = 0.0;
    float min = 10.0;
    for (int row = 0; row < numRows; row++){
        for (int col = 0; col < numColumns; col++){    
            float diff = A[row*numColumns+col] - B[row*numColumns+col];
            if (diff > max) max = diff;
            if (diff < min) min = diff;
            sum += diff;
        }
    }
    cout << "mean: " << sum / (numRows*numColumns) << " max: " << max << " min: " << min << endl;
}

// +++++++++++++++++++++++++

#define TILE_WIDTH 6

//ComputeC=A*B
__global__ void Matrix_Mul_GPU(double *A, double *B, double *C, int d1, int d2, int d3)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x; // Col
    int i = threadIdx.y + blockIdx.y * blockDim.y; // Row
    if (i < d1 && j < d3)
    {
        double s = 0;
        for (int k = 0; k < d2; k++)
        {
            s += A[i * d2 + k] * B[j + k * d3];
        }
        C[i * d3 + j] = s;
    }
}

void matrix_dot_GPU1(matrix_t *m1, matrix_t *m2, matrix_t *res)
{


    int numARows = m1->rows; // number of rows in the matrix A
    int numAColumns = m1->columns; // number of columns in the matrix A
    int numBRows = m2->rows; // number of rows in the matrix B
    int numBColumns = m2->columns; // number of columns in the matrix B
    int numCRows = res->rows; // number of rows in the matrix C
    int numCColumns = res->columns; // number of columns in the matrix C 
    assert(numAColumns == numBRows);

    // Initialize matrices on the host

    // CUDA PART
    float *deviceA;
    float *deviceB;
    float *deviceC;
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **)&deviceC, numCRows * numBColumns * sizeof(float));

    cudaMemcpy(deviceA, m1->m, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, m2->m, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(deviceC, 0, numCRows * numCColumns * sizeof(float));
    
    dim3 blockDim(6,6);
    dim3 gridDim(ceil(((float)numBColumns) / blockDim.x), ceil(((float)numARows) / blockDim.y));
    sgemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

    cudaMemcpy(res->m, deviceC, numARows * numBColumns * sizeof(float), cudaMemcpyDeviceToHost);
    // END CUDA PART

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    double *test = res->m;

    matrix_dot1(m1, m2, res);

    compareMatrix(test,res->m,res->rows,res->columns);

}

void matrix_dot_GPU2(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    int gridSize = ceil((float)res->rows * res->columns / THREADS_PER_BLOCK);
    dim3 DimGrid(gridSize, gridSize, 1);
    dim3 DimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    Matrix_Mul_GPU<<DimGrid, DimBlock>>>(m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);
    cudaDeviceSynchronize();
}


// +++++++++++++++++++++++++

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

__device__ void matrix_function_GPU(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
            (m1->rows == res->rows));

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < m1->rows * m1->columns)
    { 
        res->m[index] = f(m1->m[index]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

__device__ void matrix_scalar_GPU(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < m1->rows * m1->columns)
    { 
        res->m[index] = m1->m[index] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}