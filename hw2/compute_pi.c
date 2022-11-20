#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include <omp.h>

double compute_pi(int n_samples, int n_threads) {
	int i;
	int acc = 0;
	double x, y, z;
	
        #pragma omp parallel shared(n_samples, acc) private(i, x, y, z) num_threads(n_threads)
	{
		unsigned int seed = ((unsigned) time(NULL) & 0xFFFFFFF0) | (1 + omp_get_thread_num());

		#pragma omp for reduction(+:acc)
		for (i = 0; i < n_samples; i++) 
                {
			x = (double)rand_r(&seed) / RAND_MAX;
			y = (double)rand_r(&seed) / RAND_MAX;
			z = x * x + y * y;
			if (z <= 1) 
				acc++;
		}
	}
	
        return (4.0 * acc) / n_samples;
}

int main(int argc, char* argv[]) {
	int n_threads;
	printf("Enter number of threads: ");
	scanf("%d", &n_threads);
	printf("\n");
	
	int n_samples = 100000000;
	int n_experiments = 10;
	double time = 0.0, total_time = 0.0;
	double pi = 0.0;

	for (int i_exp = 0; i_exp < n_experiments; i_exp++) {
		time = omp_get_wtime();
		pi = compute_pi(n_samples, n_threads);
		printf("Pi = %lf\n", pi);
		total_time += omp_get_wtime() - time;
	}
	printf("Mean spent time (seconds) = %lf\n\n", total_time / n_experiments);
	
	return 0;
}
