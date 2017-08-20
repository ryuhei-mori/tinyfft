#include <complex.h>

void tfft_init_avx(int k, double complex *w);

void tfft_fft_avx(int k, double *xr, double *xi, const double complex *w);

void tfft_ifft_avx(int k, double *xr, double *xi, const double complex *w);

void tfft_convolver_avx(int k, double *xr, double *xi, const double complex *w);

