#include <complex.h>

void fft(int k, double complex *A, const double complex *w);

void ifft(int k, double complex *A, const double complex *w);

void convolver(int k, double complex *A, const double complex *w);

void init(int k, double complex *w);

