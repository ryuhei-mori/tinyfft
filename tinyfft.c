#include <complex.h>

#define PI2 6.28318530717958647692

typedef double complex cmplx;

void fft(int k, cmplx *A, const cmplx *w){
  const int m = 1 << k;
  int u = 1;
  int v = m/2;
  int i;
  for(i=k;i>0;i--){
    int jh;
    for(jh=0;jh<u;jh++){
      cmplx wj = w[jh];
      int j, je;
      for(j=jh<<i, je=j|v; j<je; j++){
        cmplx Ajv = wj * A[j|v];
        A[j|v] = A[j] - Ajv;
        A[j] += Ajv;
      }
    }
    u <<= 1;
    v >>= 1;
  }
}

void ifft(int k, cmplx *A, const cmplx *w){
  const int m = 1 << k;
  int u = m/2;
  int v = 1;
  int i;
  for(i=1;i<=k;i++){
    int jh;
    for(jh=0;jh<u;jh++){
      cmplx wj = conj(w[jh]);
      int j, je;
      for(j=jh<<i, je=j|v; j<je; j++){
        cmplx Ajv = A[j] - A[j|v];
        A[j] += A[j|v];
        A[j|v] = wj * Ajv;
      }
    }
    u >>= 1;
    v <<= 1;
  }
}


void convolver(int k, cmplx *A, const cmplx *w){
  int i;
  const int m = 1 << k;

  fft(k, A, w);
  A[0] = 4 * creal(A[0]) * cimag(A[0]) * I;
  A[1] = 4 * creal(A[1]) * cimag(A[1]) * I;
  for(i = 2; i < m; i+=2){
    int y = 1 << (31-__builtin_clz(i));
    int j = i^(y-1);
    A[i] = (A[i] + conj(A[j]))*(A[i] - conj(A[j]));
    A[j] = -conj(A[i]);
  }

  for(i = 0; i < m; i+=2){
    A[i/2] = (A[i]+A[i^1] - (A[i]-A[i^1])*w[i/2]*I)/(4*m);
  }

  ifft(k-1, A, w);
}

void genw(int i, int b, cmplx z, cmplx *w){
  if(b == 0){
    w[i] = z;
  }
  else {
    genw(i, b>>1, z, w);
    genw(i|b, b>>1, z*w[b], w);
  }
}

void init(int k, cmplx *w){
  int i, j;
  const int m = 1<<k;
  const double arg = -PI2/m;
  for(i=1, j=m/4; j; i<<=1, j>>=1){
    w[i] = cexp(I * (arg * j));
  }
  genw(0, m/4, 1, w);
}
