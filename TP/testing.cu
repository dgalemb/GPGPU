#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "mnist.h"
#include "matrix.h"
#include "ann.h"

// ---

#define CREATE_CUDAEVENT cudaEvent_t start, stop; \
cudaEventCreate(&start); \
cudaEventCreate(&stop);
#define START_CUDAEVENT cudaEventRecord(start, 0);
#define STOP_AND_PRINT_CUDAEVENT(txt) cudaEventRecord(stop, 0);\
cudaEventSynchronize(stop);\
{float elapsedTime;\
cudaEventElapsedTime(&elapsedTime, start, stop);\
printf("Time to %s %3.1f ms\n", #txt, elapsedTime);}

#define CREATE_CPUEVENT clock_t clock_start, clock_stop;\
double cpu_time;
#define START_CPUEVENT clock_start = clock();
#define STOP_AND_PRINT_CPUEVENT(txt) clock_stop = clock();\
cpu_time = ((double) (clock_stop - clock_start)) / CLOCKS_PER_SEC;\
printf("Time to %s %3.4f s\n", #txt, cpu_time);

// ---

void test_matrix_dot() {
    matrix_t *A = alloc_matrix(2, 2);
    A->m[0] = 4; A->m[1] = 2; A->m[2] = 2; A->m[3] = 4;
    print_matrix(A, false);


    matrix_t *B = alloc_matrix(2, 2);
    B->m[0] = 2; B->m[1] = 4; B->m[2] = 4; B->m[3] = 2;
    print_matrix(B, false);

    matrix_t *C = alloc_matrix(2, 2);
    matrix_dot_GPU2(A, B, C);
    print_matrix(C, false);

    matrix_dot(A, B, C);
    print_matrix(C, false);
}

void test_matrix_sum() {
    matrix_t *A = alloc_matrix(2, 2);
    A->m[0] = 4; A->m[1] = 2; A->m[2] = 2; A->m[3] = 4;
    print_matrix(A, false);

    matrix_t *B = alloc_matrix(2, 2);
    B->m[0] = 2; B->m[1] = 4; B->m[2] = 4; B->m[3] = 2;
    print_matrix(B, false);

    matrix_t *C = alloc_matrix(2, 2);
    matrix_sum_GPU(A, B, C);
    print_matrix(C, false);
}

void test_matrix_minus() {
    matrix_t *A = alloc_matrix(2, 2);
    A->m[0] = 4; A->m[1] = 2; A->m[2] = 2; A->m[3] = 4;
    print_matrix(A, false);

    matrix_t *B = alloc_matrix(2, 2);
    B->m[0] = 2; B->m[1] = 4; B->m[2] = 4; B->m[3] = 2;
    print_matrix(B, false);

    matrix_t *C = alloc_matrix(2, 2);
    matrix_minus_GPU(A, B, C);
    print_matrix(C, false);
}

void test_matrix_scalar() {
    matrix_t *A = alloc_matrix(2, 2);
    A->m[0] = 4; A->m[1] = 2; A->m[2] = 2; A->m[3] = 4;
    print_matrix(A, false);

    double B = 5.0;

    matrix_t *C = alloc_matrix(2, 2);
    matrix_scalar_GPU(A, B, C);
    print_matrix(C, false);

    matrix_t *C = alloc_matrix(2, 2);
    matrix_scalar(A, B, C);
    print_matrix(C, false);
}

int main(int argc, char *argv[])
{
    test_matrix_dot();
    test_matrix_sum();
    test_matrix_minus();
    test_matrix_scalar();
    return 0;
}