# Tiny FFT
A tiny implementation of Fast Fourier Transform for sizes power of 2

    clang -O3 tinyfft.c -c

## List of provided functions

    void init(int k, double complex *w);

Initializing a table w of twiddle factors for FFT of length 2^k.

    void fft(int k, double complex *x, const double complex *w);

A complex vector x of length 2^k is Fourier transformed by using a table w. After the transform, elements are in bitreversal order.

    void ifft(int k, double complex *x, const double complex *w);

A complex vector x of length 2^k is inverse Fourier transformed by using a table w. Before the transform, elements are assumed to be in bitreversal order.

    void convolver(int k, double complex *x, const double complex *w);

Real vectors embedded in real and complex parts of x are convolved.


