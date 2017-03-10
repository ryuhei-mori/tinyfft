#include <complex.h>

void tfft_init(int k, double complex *w);

void tfft_fft(int k, double complex *A, const double complex *w);

void tfft_ifft(int k, double complex *A, const double complex *w);

void tfft_convolver(int k, double complex *A, const double complex *w);

