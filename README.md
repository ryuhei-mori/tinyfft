# Tiny FFT
A tiny implementation of the fast Fourier transform for sizes power of 2.

    clang -O3 tinyfft.c -c
    clang -O3 example.c tinyfft.o -lm

## List of provided functions

    void init(int k, double complex *w);

A table w of twiddle factors is initialized for FFT of length 2^k. The size of table is 2^(k-1).

    void fft(int k, double complex *x, const double complex *w);

A complex vector x of length 2^k is Fourier transformed by using a table w. After the transform, elements are in the bitreversal order.

    void ifft(int k, double complex *x, const double complex *w);

A complex vector x of length 2^k is inverse Fourier transformed by using a table w. Before the transform, elements are assumed to be in the bitreversal order.

    void convolver(int k, double complex *x, const double complex *w);

Real vectors embedded in real and complex parts of x are convolved.

## Performance
Empirically, for large vectors, Tiny FFT is 2x slower than FFTW3.

