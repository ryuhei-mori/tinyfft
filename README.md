# Tiny FFT
A tiny implementation of the fast Fourier transform for sizes power of 2.

    clang -O3 example.c tinyfft.c -lm

There is also an implmentation using AVX whose performance is comparable to FFTW3 for some length (empirically 2^17 to 2^20).

    clang -O3 example.c tinyfft_avx.c -lm -march=native

## List of provided functions in 'tinyfft.c'

    void tfft_init(int k, double complex *w);

A table w of twiddle factors is initialized for FFT of length 2^k. The size of table `w` is 2^(k-1).

    void tfft_fft(int k, double complex *x, const double complex *w);

A complex vector `x` of length 2^k is Fourier transformed by using a table `w`. After the transform, elements are placed in the bitreversal order.

    void tfft_ifft(int k, double complex *x, const double complex *w);

A complex vector `x` of length 2^k is inverse Fourier transformed by using a table `w`. The elements of `x` are assumed to be placed in the bitreversal order.

    void tfft_convolver(int k, double complex *x, const double complex *w);

Two real vectors embedded in real and complex parts of `x` are cyclically convolved.

## List of provided functions in 'tinyfft_avx.c'

    void tfft_init_avx(int k, double complex *w);

A table w of twiddle factors is initialized for FFT of length 2^k. The size of table `w` is 2^(k-1).

    void tfft_fft_avx(int k, double *xr, double *xi, const double complex *w);

A complex vector `(xr, xi)` of length 2^k is Fourier transformed by using a table `w`. After the transform, elements are placed in the bitreversal order.

    void tfft_ifft_avx(int k, double *xr, double *xi, const double complex *w);

A complex vector `(xr, xi)` of length 2^k is inverse Fourier transformed by using a table `w`. The elements of `xr` and `xi` are assumed to be placed in the bitreversal order.

    void tfft_convolver_avx(int k, double *xr, double *xi, const double complex *w);

Two real vectors `xr` and `xi` are cyclically convolved.

## Performance
Empirically, for vectors of length between 2^17 and 2^20, Tiny FFT is faster than FFTW3 with FFTW_MEASURE.
For other cases, Tiny FFT is slower than FFTW3.

