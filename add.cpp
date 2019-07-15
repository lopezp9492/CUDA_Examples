//Example from Nvidia DevBlogs
//https://devblogs.nvidia.com/even-easier-introduction-cuda/

#include <iostream>
#include <math.h>

//function to add elements of 2 arrays
void add(int n, float *x, float *y
{
	for (int i = 0 i < n , i++)
	{
		y[i] = x[i] + y[i];
	}
}


int main(void)
{
	int N = 1 << 20; //1Million  elements

	float *x = new float[N];
	float *y = new float[N];

	//initiate arrays 
	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	//Run function on 1M elements on the CPU
	add(N, x, y);

	//Check for errors
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::out << "Max error: " << maxError << std::endl;

	//Free memory
	delete [] x;
	delete [] y;

	return 0;

}