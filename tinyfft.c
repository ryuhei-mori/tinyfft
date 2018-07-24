#include <complex.h>

#define PI2 6.28318530717958647692

typedef double complex cmplx;

void tfft_genw(int i, int b, cmplx z, cmplx *w){
  if(b == 0)
    w[i] = z;
  else {
    tfft_genw(i, b>>1, z, w);
    tfft_genw(i|b, b>>1, z*w[b], w);
  }
}

void tfft_init(int k, cmplx *w){
  int i, j;
  const int m = 1<<k;
  const double arg = -PI2/m;
  for(i=1, j=m/4; j; i<<=1, j>>=1){
    w[i] = cexp(I * (arg * j));
  }
  tfft_genw(0, m/4, 1, w);
}

void tfft_fft(int k, cmplx *A, const cmplx *w){
  const int m = 1 << k;
  int u = 1;
  int v = m/4;
  int i, j;
  if(k&1){
    for(j=0; j<m/2; j++){
      cmplx Ajv = A[j+(m/2)];
      A[j+(m/2)] = A[j] - Ajv;
      A[j] += Ajv;
    }
    u <<= 1;
    v >>= 1;
  }
  for(i=k&~1;i>0;i-=2){
    int jh;
    for(jh=0;jh<u;jh++){
      cmplx wj = w[jh<<1];
      cmplx wj2 = w[jh];
      cmplx wj3 = wj2 * wj;
      int je;
      for(j = jh << i, je = j+v;j<je; j++){
        cmplx tmp0 = A[j];
        cmplx tmp1 = wj * A[j+v];
        cmplx tmp2 = wj2 * A[j+2*v];
        cmplx tmp3 = wj3 * A[j+3*v];

        cmplx ttmp0 = tmp0 + tmp2;
        cmplx ttmp2 = tmp0 - tmp2;
        cmplx ttmp1 = tmp1 + tmp3;
        cmplx ttmp3 = -I * (tmp1 - tmp3);

        A[j] = ttmp0 + ttmp1;
        A[j+v] = ttmp0 - ttmp1;
        A[j+2*v] = ttmp2 + ttmp3;
        A[j+3*v] = ttmp2 - ttmp3;
      }
    }
    u <<= 2;
    v >>= 2;
  }
}

void tfft_ifft(int k, cmplx *A, const cmplx *w){
  const int m = 1 << k;
  int u = m/4;
  int v = 1;
  int i, j;
  for(i=2;i<=k;i+=2){
    int jh;
    for(jh=0;jh<u;jh++){
      cmplx wj = conj(w[jh<<1]);
      cmplx wj2 = conj(w[jh]);
      cmplx wj3 = wj2 * wj;
      int je;
      for(j = jh << i, je = j+v;j<je; j++){
        cmplx tmp0 = A[j];
        cmplx tmp1 = A[j+v];
        cmplx tmp2 = A[j+2*v];
        cmplx tmp3 = A[j+3*v];

        cmplx ttmp0 = tmp0 + tmp1;
        cmplx ttmp1 = tmp0 - tmp1;
        cmplx ttmp2 = tmp2 + tmp3;
        cmplx ttmp3 = I * (tmp2 - tmp3);

        A[j] = ttmp0 + ttmp2;
        A[j+v] = wj * (ttmp1 + ttmp3);
        A[j+2*v] = wj2 * (ttmp0 - ttmp2);
        A[j+3*v] = wj3 * (ttmp1 - ttmp3);
      }
    }
    u >>= 2;
    v <<= 2;
  }
  if(k&1){
    for(j = 0;j<m/2; j++){
      cmplx Ajv = A[j+(m/2)];
      A[j+(m/2)] = A[j] - Ajv;
      A[j] += Ajv;
    }
  }
}

void tfft_convolver(int k, cmplx *A, const cmplx *w){
  int i, y;
  const int m = 1 << k;

  tfft_fft(k, A, w);
  A[0] = 4 * creal(A[0]) * cimag(A[0]) * I;
  A[1] = 4 * creal(A[1]) * cimag(A[1]) * I;
  i = 2;
  for(y = 2; y < m; y <<= 1){
    for(; i < 2*y; i+=2){
      int j = i^(y-1);
      A[i] = (A[i] + conj(A[j]))*(A[i] - conj(A[j]));
      A[j] = -conj(A[i]);
    }
  }

  for(i = 0; i < m; i+=2){
    A[i/2] = (-(A[i]+A[i^1])*I + (A[i]-A[i^1])*conj(w[i/2]))/(4*m);
  }

  tfft_ifft(k-1, A, w);
}

