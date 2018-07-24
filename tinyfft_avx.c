#include <complex.h>
#include <immintrin.h>

#define PI2 6.28318530717958647692

typedef double complex cmplx;

void tfft_genw_avx(int i, int b, cmplx z, cmplx *w){
  if(b == 0)
    w[i] = z;
  else {
    tfft_genw_avx(i, b>>1, z, w);
    tfft_genw_avx(i|b, b>>1, z*w[b], w);
  }
}

void tfft_init_avx(int k, cmplx *w){
  int i, j;
  const int m = 1<<k;
  const double arg = -PI2/m;
  for(i=1, j=m/4; j; i<<=1, j>>=1){
    w[i] = cexp(I * (arg * j));
  }
  tfft_genw_avx(0, m/4, 1, w);
}

void complex_mul_avx(__m256d *ar, __m256d *ai, __m256d *br, __m256d *bi){
/*
  __m256d arbr = _mm256_mul_pd(*ar, *br);
  __m256d aibi = _mm256_mul_pd(*ai, *bi);
  __m256d arbi = _mm256_mul_pd(*ar, *bi);
  __m256d aibr = _mm256_mul_pd(*ai, *br);
  *ar = _mm256_sub_pd(arbr, aibi);
  *ai = _mm256_add_pd(arbi, aibr);
*/
  __m256d aibi = _mm256_mul_pd(*ai, *bi);
  __m256d aibr = _mm256_mul_pd(*ai, *br);
  *ai = _mm256_fmadd_pd(*ar, *bi, aibr);
  *ar = _mm256_fmsub_pd(*ar, *br, aibi);
}

void complex_mul(double *ar, double *ai, double br, double bi){
  double tmpar = *ar * br - *ai * bi;
  *ai = *ar * bi + *ai * br;
  *ar = tmpar;
}


void tfft_fft_avx(int k, double *xr, double *xi,  const cmplx *w){
  const int m = 1 << k;
  int u = 1;
  int v = m/4;
  int i, j;

  if(k&1){
    for(j=0; j<m/2; j+=4){
      __m256d ar = _mm256_load_pd(xr+j);
      __m256d ai = _mm256_load_pd(xi+j);
      __m256d br = _mm256_load_pd(xr+j+m/2);
      __m256d bi = _mm256_load_pd(xi+j+m/2);
      _mm256_store_pd(xr+j, _mm256_add_pd(ar, br));
      _mm256_store_pd(xi+j, _mm256_add_pd(ai, bi));
      _mm256_store_pd(xr+j+m/2, _mm256_sub_pd(ar, br));
      _mm256_store_pd(xi+j+m/2, _mm256_sub_pd(ai, bi));
    }
    u <<= 1;
    v >>= 1;
  }

  for(i=k&~1;v>=4;i-=2){
    int jh;
    for(jh=0;jh<u;jh++){
      double wjr = creal(w[jh<<1]);
      double wji = cimag(w[jh<<1]);
      double wj2r = creal(w[jh]);
      double wj2i = cimag(w[jh]);
      double wj3r = wjr*wj2r-wji*wj2i;
      double wj3i = wjr*wj2i+wji*wj2r;
      int je;
        __m256d wjr4 = _mm256_broadcast_sd(& wjr);
        __m256d wji4 = _mm256_broadcast_sd(& wji);
        __m256d wj2r4 = _mm256_broadcast_sd(& wj2r);
        __m256d wj2i4 = _mm256_broadcast_sd(& wj2i);
        __m256d wj3r4 = _mm256_broadcast_sd(& wj3r);
        __m256d wj3i4 = _mm256_broadcast_sd(& wj3i);
        for(j = jh << i, je = j+v;j<je; j+=4){
          __m256d tmp0r = _mm256_load_pd(xr+j);
          __m256d tmp0i = _mm256_load_pd(xi+j);
          __m256d tmp1r = _mm256_load_pd(xr+v+j);
          __m256d tmp1i = _mm256_load_pd(xi+v+j);
          complex_mul_avx(&tmp1r, &tmp1i, &wjr4, &wji4);
          __m256d tmp2r = _mm256_load_pd(xr+2*v+j);
          __m256d tmp2i = _mm256_load_pd(xi+2*v+j);
          complex_mul_avx(&tmp2r, &tmp2i, &wj2r4, &wj2i4);
          __m256d tmp3r = _mm256_load_pd(xr+3*v+j);
          __m256d tmp3i = _mm256_load_pd(xi+3*v+j);
          complex_mul_avx(&tmp3r, &tmp3i, &wj3r4, &wj3i4);

          __m256d ttmp0r = _mm256_add_pd(tmp0r, tmp2r);
          __m256d ttmp0i = _mm256_add_pd(tmp0i, tmp2i);
          __m256d ttmp2r = _mm256_sub_pd(tmp0r, tmp2r);
          __m256d ttmp2i = _mm256_sub_pd(tmp0i, tmp2i);
          __m256d ttmp1r = _mm256_add_pd(tmp1r, tmp3r);
          __m256d ttmp1i = _mm256_add_pd(tmp1i, tmp3i);
          __m256d ttmp3r = _mm256_sub_pd(tmp1i, tmp3i);
          __m256d ttmp3i = _mm256_sub_pd(-tmp1r, -tmp3r);

          _mm256_store_pd(xr+j, _mm256_add_pd(ttmp0r, ttmp1r));
          _mm256_store_pd(xi+j, _mm256_add_pd(ttmp0i, ttmp1i));
          _mm256_store_pd(xr+j+v, _mm256_sub_pd(ttmp0r, ttmp1r));
          _mm256_store_pd(xi+j+v, _mm256_sub_pd(ttmp0i, ttmp1i));
          _mm256_store_pd(xr+j+2*v, _mm256_add_pd(ttmp2r, ttmp3r));
          _mm256_store_pd(xi+j+2*v, _mm256_add_pd(ttmp2i, ttmp3i));
          _mm256_store_pd(xr+j+3*v, _mm256_sub_pd(ttmp2r, ttmp3r));
          _mm256_store_pd(xi+j+3*v, _mm256_sub_pd(ttmp2i, ttmp3i));
      }
    }
    u <<= 2;
    v >>= 2;
  }
  for(;i>0;i-=2){
    int jh;
    for(jh=0;jh<u;jh++){
      double wjr = creal(w[jh<<1]);
      double wji = cimag(w[jh<<1]);
      double wj2r = creal(w[jh]);
      double wj2i = cimag(w[jh]);
      double wj3r = wjr*wj2r-wji*wj2i;
      double wj3i = wjr*wj2i+wji*wj2r;
      int je;
      for(j = jh << i, je = j+v;j<je; j++){
        double tmp0r = xr[j];
        double tmp0i = xi[j];
        double tmp1r = xr[j+v];
        double tmp1i = xi[j+v];
        complex_mul(&tmp1r, &tmp1i, wjr, wji);
        double tmp2r = xr[j+2*v];
        double tmp2i = xi[j+2*v];
        complex_mul(&tmp2r, &tmp2i, wj2r, wj2i);
        double tmp3r = xr[j+3*v];
        double tmp3i = xi[j+3*v];
        complex_mul(&tmp3r, &tmp3i, wj3r, wj3i);
  
        double ttmp0r = tmp0r + tmp2r;
        double ttmp0i = tmp0i + tmp2i;
        double ttmp2r = tmp0r - tmp2r;
        double ttmp2i = tmp0i - tmp2i;
        double ttmp1r = tmp1r + tmp3r;
        double ttmp1i = tmp1i + tmp3i;
        double ttmp3r = tmp1i - tmp3i;
        double ttmp3i = -(tmp1r - tmp3r);

        xr[j] = ttmp0r + ttmp1r;
        xi[j] = ttmp0i + ttmp1i;
        xr[j+v] = ttmp0r - ttmp1r;
        xi[j+v] = ttmp0i - ttmp1i;
        xr[j+2*v] = ttmp2r + ttmp3r;
        xi[j+2*v] = ttmp2i + ttmp3i;
        xr[j+3*v] = ttmp2r - ttmp3r;
        xi[j+3*v] = ttmp2i - ttmp3i;
      }
    }
    u <<= 2;
    v >>= 2;
  }
}


void tfft_ifft_avx(int k, double *xr, double *xi,  const cmplx *w){
  const int m = 1 << k;
  int u = m/4;
  int v = 1;
  int i, j;


  if(k>=2){
    int jh;
    for(jh=0;jh<u;jh++){
      double wjr = creal(w[jh<<1]);
      double wji = -cimag(w[jh<<1]);
      double wj2r = creal(w[jh]);
      double wj2i = -cimag(w[jh]);
      double wj3r = wjr*wj2r-wji*wj2i;
      double wj3i = wjr*wj2i+wji*wj2r;
      j = jh << 2;
        double tmp0r = xr[j];
        double tmp0i = xi[j];
        double tmp1r = xr[j+v];
        double tmp1i = xi[j+v];
        double tmp2r = xr[j+2*v];
        double tmp2i = xi[j+2*v];
        double tmp3r = xr[j+3*v];
        double tmp3i = xi[j+3*v];
  
        double ttmp0r = tmp0r + tmp1r;
        double ttmp0i = tmp0i + tmp1i;
        double ttmp1r = tmp0r - tmp1r;
        double ttmp1i = tmp0i - tmp1i;
        double ttmp2r = tmp2r + tmp3r;
        double ttmp2i = tmp2i + tmp3i;
        double ttmp3r = -(tmp2i - tmp3i);
        double ttmp3i = tmp2r - tmp3r;

        xr[j] = ttmp0r + ttmp2r;
        xi[j] = ttmp0i + ttmp2i;
        xr[j+v] = ttmp1r + ttmp3r;
        xi[j+v] = ttmp1i + ttmp3i;
        xr[j+2*v] = ttmp0r - ttmp2r;
        xi[j+2*v] = ttmp0i - ttmp2i;
        xr[j+3*v] = ttmp1r - ttmp3r;
        xi[j+3*v] = ttmp1i - ttmp3i;

        complex_mul(xr+j+v, xi+j+v, wjr, wji);
        complex_mul(xr+j+2*v, xi+j+2*v, wj2r, wj2i);
        complex_mul(xr+j+3*v, xi+j+3*v, wj3r, wj3i);
    }
    u >>= 2;
    v <<= 2;
  }

  for(i=4;i<=k;i+=2){
    int jh;
    for(jh=0;jh<u;jh++){
      double wjr = creal(w[jh<<1]);
      double wji = -cimag(w[jh<<1]);
      double wj2r = creal(w[jh]);
      double wj2i = -cimag(w[jh]);
      double wj3r = wjr*wj2r-wji*wj2i;
      double wj3i = wjr*wj2i+wji*wj2r;
      int je;
        __m256d wjr4 = _mm256_broadcast_sd(& wjr);
        __m256d wji4 = _mm256_broadcast_sd(& wji);
        __m256d wj2r4 = _mm256_broadcast_sd(& wj2r);
        __m256d wj2i4 = _mm256_broadcast_sd(& wj2i);
        __m256d wj3r4 = _mm256_broadcast_sd(& wj3r);
        __m256d wj3i4 = _mm256_broadcast_sd(& wj3i);
        for(j = jh << i, je = j+v;j<je; j+=4){
          __m256d tmp0r = _mm256_load_pd(xr+j);
          __m256d tmp0i = _mm256_load_pd(xi+j);
          __m256d tmp1r = _mm256_load_pd(xr+v+j);
          __m256d tmp1i = _mm256_load_pd(xi+v+j);
          __m256d tmp2r = _mm256_load_pd(xr+2*v+j);
          __m256d tmp2i = _mm256_load_pd(xi+2*v+j);
          __m256d tmp3r = _mm256_load_pd(xr+3*v+j);
          __m256d tmp3i = _mm256_load_pd(xi+3*v+j);

          __m256d ttmp0r = _mm256_add_pd(tmp0r, tmp1r);
          __m256d ttmp0i = _mm256_add_pd(tmp0i, tmp1i);
          __m256d ttmp1r = _mm256_sub_pd(tmp0r, tmp1r);
          __m256d ttmp1i = _mm256_sub_pd(tmp0i, tmp1i);
          __m256d ttmp2r = _mm256_add_pd(tmp2r, tmp3r);
          __m256d ttmp2i = _mm256_add_pd(tmp2i, tmp3i);
          __m256d ttmp3r = _mm256_sub_pd(-tmp2i, -tmp3i);
          __m256d ttmp3i = _mm256_sub_pd(tmp2r, tmp3r);

          __m256d tttmp0r = _mm256_add_pd(ttmp0r, ttmp2r);
          __m256d tttmp0i = _mm256_add_pd(ttmp0i, ttmp2i);
          __m256d tttmp2r = _mm256_sub_pd(ttmp0r, ttmp2r);
          __m256d tttmp2i = _mm256_sub_pd(ttmp0i, ttmp2i);
          __m256d tttmp1r = _mm256_add_pd(ttmp1r, ttmp3r);
          __m256d tttmp1i = _mm256_add_pd(ttmp1i, ttmp3i);
          __m256d tttmp3r = _mm256_sub_pd(ttmp1r, ttmp3r);
          __m256d tttmp3i = _mm256_sub_pd(ttmp1i, ttmp3i);

          complex_mul_avx(&tttmp1r, &tttmp1i, &wjr4, &wji4);
          complex_mul_avx(&tttmp2r, &tttmp2i, &wj2r4, &wj2i4);
          complex_mul_avx(&tttmp3r, &tttmp3i, &wj3r4, &wj3i4);

          _mm256_store_pd(xr+j, tttmp0r);
          _mm256_store_pd(xi+j, tttmp0i);
          _mm256_store_pd(xr+j+v, tttmp1r);
          _mm256_store_pd(xi+j+v, tttmp1i);
          _mm256_store_pd(xr+j+2*v, tttmp2r);
          _mm256_store_pd(xi+j+2*v, tttmp2i);
          _mm256_store_pd(xr+j+3*v, tttmp3r);
          _mm256_store_pd(xi+j+3*v, tttmp3i);
      }
    }
    u >>= 2;
    v <<= 2;
  }
  if(k&1){
    for(j=0; j<m/2; j+=4){
      __m256d ar = _mm256_load_pd(xr+j);
      __m256d ai = _mm256_load_pd(xi+j);
      __m256d br = _mm256_load_pd(xr+j+m/2);
      __m256d bi = _mm256_load_pd(xi+j+m/2);
      _mm256_store_pd(xr+j, _mm256_add_pd(ar, br));
      _mm256_store_pd(xi+j, _mm256_add_pd(ai, bi));
      _mm256_store_pd(xr+j+m/2, _mm256_sub_pd(ar, br));
      _mm256_store_pd(xi+j+m/2, _mm256_sub_pd(ai, bi));
    }
 
/*
    for(j=0; j<m/2; j++){
      double tmpr = xr[j+(m/2)];
      double tmpi = xi[j+(m/2)];
      xr[j+(m/2)] = xr[j] - tmpr;
      xi[j+(m/2)] = xi[j] - tmpi;
      xr[j] += tmpr;
      xi[j] += tmpi;
    }
*/
  }
}

void tfft_convolver_avx(int k, double *xr, double *xi, const cmplx *w){
  int i, y;
  const int m = 1 << k;

  tfft_fft_avx(k, xr, xi, w);
  xi[0] = 4 * xr[0] * xi[0];
  xi[1] = 4 * xr[1] * xi[1];
  xr[0] = xr[1] = 0;
  i = 2;
  for(y = 2; y < m; y <<= 1){
    for(; i < 2*y; i+=2){
      int j = i^(y-1);
      double tmpr = xr[i] - xr[j];
      double tmpi = xi[i] + xi[j];
      xr[i] += xr[j];
      xi[i] -= xi[j];

      complex_mul(xr+i, xi+i, tmpr, tmpi);

      xr[j] = -xr[i];
      xi[j] = xi[i];
    }
  }

  for(i = 0; i < m; i+=2){
    double ttmpr = xr[i] + xr[i^1];
    double ttmpi = xi[i] + xi[i^1];
    double tmpr = xr[i] - xr[i^1];
    double tmpi = xi[i] - xi[i^1];
    complex_mul(&tmpr, &tmpi, creal(w[i/2]), -cimag(w[i/2]));
    xr[i/2] = (ttmpi + tmpr) / (4*m);
    xi[i/2] = (-ttmpr + tmpi) / (4*m);
  }

  tfft_ifft_avx(k-1, xr, xi, w);
}
