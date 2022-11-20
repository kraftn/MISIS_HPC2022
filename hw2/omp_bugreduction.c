#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

float dotprod(float * a, float * b, size_t N, float * sum)
{
    int i, tid;

    tid = omp_get_thread_num();

    #pragma omp for reduction(+:sum[:1])
    for (i = 0; i < N; ++i)
    {
        *sum += a[i] * b[i];
        #pragma omp critical
        printf("tid = %d i = %d\n", tid, i);
    }
		
    return *sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 100;
    int i;
    float sum;
    float a[N], b[N];


    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (double)i;
    }

    sum = 0.0;    

    #pragma omp parallel shared(a, b, sum) firstprivate(N)
    dotprod(a, b, N, &sum);

    printf("OMP sum = %f\n",sum);

    sum = 0.0;
    for (i = 0; i < N; ++i)
    {
	sum += a[i] * b[i];
    }

    printf("True sum = %f\n", sum);

    return 0;
}
