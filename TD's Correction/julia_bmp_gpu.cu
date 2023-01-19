/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

 #include <iostream>
 #include "bitmap_image.hpp"

#define DIM 1000

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__
int julia( int x, int y ) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__
void kernel( unsigned char *ptr ){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < DIM && y < DIM){
        int offset = x + y * DIM;

        int juliaValue = julia( x, y );
        ptr[offset*3 + 0] = 255 * juliaValue;
        ptr[offset*3 + 1] = 0;
        ptr[offset*3 + 2] = 0;
    }        
 }

int main( void ) {
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int size = DIM*DIM*3;
    unsigned char *h_ptr = (unsigned char *)malloc(size);
    unsigned char *d_ptr;

    cudaMalloc((void**)&d_ptr, size);
    
    cudaEventRecord(start, 0);
    dim3 DimGrid ( ceil(DIM/16.0) , ceil(DIM/16.0)  , 1);
    dim3 DimBlock ( 16 , 16 , 1);

    kernel<<< DimGrid , DimBlock >>>( d_ptr );
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to compute %3.1f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy( h_ptr, d_ptr, size, cudaMemcpyDeviceToHost );
    cudaFree(d_ptr);
	// Write BMP image
    bitmap_image img(DIM,DIM);
    img.clear();
    for (int y = DIM-1; y >= 0; y--)
	{
		for (int x = DIM-1; x >= 0; x--)
		{
            int offset = x + y * DIM;
            img.set_pixel(x, y, h_ptr[offset*3], h_ptr[offset*3+1], h_ptr[offset*3+2]);            
        }
    }
    img.save_image("test.bmp");
	return 0;
}

