/*
* Copyright 2012 NVIDIA Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

__device__ float atomicMax(float* const address, const float val)
{
	if ( *address >= val )
		return *address;
	int* const address_as_i = (int*)address;
	int old = *address_as_i;
	int assumed = old;
	do {
		assumed = old;
		if ( __int_as_float(assumed) >= val )
			break;
		old = atomicCAS(address_as_i, assumed, __float_as_int(val) );
	} while (assumed != old);
	return __int_as_float(old);
}

__global__ void jacobi_kernel( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d )
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	float residue = 0.0f;
	if ( j >= 1 && j < n-1 && i >= 1 && i < m-1 )
	{
		Anew_d[j *m+ i] = 0.25f * ( A_d[j     *m+ (i+1)] + A_d[j     *m+ (i-1)]
							   +    A_d[(j-1) *m+ i]     + A_d[(j+1) *m+ i]);
		residue = fabsf(Anew_d[j *m+ i]-A_d[j *m+ i]);
		atomicMax( residue_d, residue );
	}
}

__global__ void copy_kernel( float* const A_d, const float* const Anew_d, const int n, const int m )
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if ( j >= 1 && j < n-1 && i >= 1 && i < m-1 )
	{
		A_d[j *m + i] = Anew_d[j *m + i];
	}
}

extern "C" float launch_jacobi_kernel( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d )
{
	const dim3 dimBlock(16,16,1);
	const dim3 dimGrid((m/dimBlock.x)+1,(n/dimBlock.y)+1,1);
	float residue = 0.f;
	cudaMemcpy( residue_d, &residue, sizeof(float), cudaMemcpyHostToDevice );

	jacobi_kernel<<<dimGrid,dimBlock>>>(A_d,Anew_d,n,m,residue_d);
	
	cudaMemcpy( &residue, residue_d, sizeof(float), cudaMemcpyDeviceToHost );
	return residue;
}

extern "C" void launch_jacobi_kernel_async( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d  )
{
	const dim3 dimBlock(16,16,1);
	const dim3 dimGrid((m/dimBlock.x)+1,(n/dimBlock.y)+1,1);
	float residue = 0.f;
	cudaMemcpy( residue_d, &residue, sizeof(float), cudaMemcpyHostToDevice );
	if ( n > 0 && m > 0 )
	{
		jacobi_kernel<<<dimGrid,dimBlock>>>(A_d,Anew_d,n,m,residue_d);
	}
}

extern "C" float wait_jacobi_kernel( const float* const residue_d )
{
	float residue = 0.f;
	cudaMemcpy( &residue, residue_d, sizeof(float), cudaMemcpyDeviceToHost );
	return residue;
}

extern "C" void launch_copy_kernel( float* const A_d, const float* const Anew_d, const int n, const int m )
{
	const dim3 dimBlock(16,16,1);
	const dim3 dimGrid((m/dimBlock.x)+1,(n/dimBlock.y)+1,1);
	copy_kernel<<<dimGrid,dimBlock>>>(A_d,Anew_d,n,m);
}

