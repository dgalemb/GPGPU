 #include <iostream>
 #include <math.h>
 #include <assert.h>
 
 using namespace std;

void initMatrix(float *m, int numRows, int numCols);
void computeMatrixMulCPU(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns);
void compareMatrix(float *A, float *B, int numRows, int numColumns);

#define CREATE_CUDAEVENT cudaEvent_t start, stop; \
cudaEventCreate(&start); \
cudaEventCreate(&stop);
#define START_CUDAEVENT cudaEventRecord(start, 0);
#define STOP_AND_PRINT_CUDAEVENT(txt) cudaEventRecord(stop, 0);\
cudaEventSynchronize(stop);\
{float elapsedTime;\
cudaEventElapsedTime(&elapsedTime, start, stop);\
printf("Time to %s %3.1f ms\n", #txt, elapsedTime);}

#define TILE_WIDTH 16

//ComputeC=A*B
__global__ 
void sgemm(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
    row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
        if (row < numARows && m * TILE_WIDTH + tx < numAColumns)
            ds_M[ty][tx] = A[row * numAColumns + m * TILE_WIDTH + tx];
        else
            ds_M[ty][tx] = 0;
        if (col < numBColumns && m * TILE_WIDTH + ty < numBRows)
            ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + col];
        else
            ds_N[ty][tx] = 0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_M[ty][k] * ds_N[k][tx];
        __syncthreads();
    }
    if (row < numARows && col < numBColumns)
        C[row * numBColumns + col] = Pvalue;

    
    /*int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < numARows && col < numBColumns) {
        float sum = 0;
        for (int ii = 0; ii < numAColumns; ii++) {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }*/
}

int main(int argc, char *argv[])
{

    CREATE_CUDAEVENT

    int numARows = atoi(argv[1]); // number of rows in the matrix A
    int numAColumns = atoi(argv[2]); // number of columns in the matrix A
    int numBRows = atoi(argv[3]); // number of rows in the matrix B
    int numBColumns = atoi(argv[4]); // number of columns in the matrix B
    int numCRows = numARows; // number of rows in the matrix C
    int numCColumns = numBColumns; // number of columns in the matrix C 
    assert(numAColumns == numBRows);

    float *A = (float *)malloc(numARows*numAColumns*sizeof(float));
    float *B = (float *)malloc(numBRows*numBColumns*sizeof(float));
    float *C = (float *)malloc(numCRows*numCColumns*sizeof(float));
    float *hostC = (float *)malloc(numCRows*numCColumns*sizeof(float));

    // Initialize matrices on the host
    initMatrix(A, numARows, numAColumns);
    initMatrix(B, numBRows, numBColumns);

    START_CUDAEVENT
    computeMatrixMulCPU(A, B, C, numARows, numAColumns, numBRows, numBColumns);
    STOP_AND_PRINT_CUDAEVENT(compute CPU)

    // CUDA PART
    float *deviceA;
    float *deviceB;
    float *deviceC;
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **)&deviceC, numCRows * numBColumns * sizeof(float));

    cudaMemcpy(deviceA, A, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(deviceC, 0, numCRows * numCColumns * sizeof(float));
    
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)numBColumns) / blockDim.x), ceil(((float)numARows) / blockDim.y));
    START_CUDAEVENT
    sgemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    STOP_AND_PRINT_CUDAEVENT(compute GPU)

    cudaMemcpy(hostC, deviceC, numARows * numBColumns * sizeof(float), cudaMemcpyDeviceToHost);
    // END CUDA PART

    compareMatrix(C, hostC, numCRows, numCColumns);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    free(A);
    free(B);
    free(C);

    return 0;
}

void initMatrix(float *m, int numRows, int numCols){
    for (int i=0; i<numRows; i++){
        for (int j=0; j<numCols; j++){
            m[i*numCols+j] = sin(i*numCols+j);
        }
    }
}

void computeMatrixMulCPU(
    float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns
){
    for (int row = 0; row < numARows; row++){
        for (int col = 0; col < numBColumns; col++){   
            C[row * numBColumns + col] = 0.0;         
            for (int n = 0; n < numAColumns; n++){
                C[row * numBColumns + col] += A[row * numAColumns + n] * B[n * numBColumns + col];
            }
        }
    }
}

void compareMatrix(float *A, float *B, int numRows, int numColumns){
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