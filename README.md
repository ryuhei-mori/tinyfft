# Tiny FFT
A tiny implementation of the fast Fourier transform for sizes power of 2.

    clang -O3 example.c tinyfft.c -lm

## List of provided functions

    void tfft_init(int k, double complex *w);

A table w of twiddle factors is initialized for FFT of length 2^k. The size of table `w` is 2^(k-1).

    void tfft_fft(int k, double complex *x, const double complex *w);

A complex vector x of length 2^k is Fourier transformed by using a table w. After the transform, elements are placed in the bitreversal order.

    void tfft_ifft(int k, double complex *x, const double complex *w);

A complex vector x of length 2^k is inverse Fourier transformed by using a table w. The elements of `x` are assumed to be placed in the bitreversal order.

    void tfft_convolver(int k, double complex *x, const double complex *w);

Two real vectors embedded in real and complex parts of x are convolved.

## Performance
For short vectors (up to length 2^17), Tiny FFT is slower than FFTW3.
Empirically, for large vectors (>= 2^18), Tiny FFT is faster than FFTW3 with FFTW_MEASURE.

