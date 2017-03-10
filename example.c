#include "tinyfft.h"
#include <stdio.h>

#define K 18

const int m = 1<<K;

double complex w[1<<(K-1)];
double complex A[1<<K];

int main(){
  int n, i;
  scanf("%d", &n);
  for(i=1; i<=n; i++){
    int a, b;
    scanf("%d%d", &a, &b);
    A[i] = a + I*b;
  }
  tfft_init(K, w);

  tfft_fft(K, A, w);
  for(i=0; i<2*n; i++){
    printf("%f %f\n", creal(A[i]), cimag(A[i]));
  }
  tfft_ifft(K, A, w);
  puts("");
  for(i=0; i<2*n; i++){
    A[i] /= m;
    printf("%f %f\n", creal(A[i]), cimag(A[i]));
  }

  tfft_convolver(K, A, w);
  for(i=0; i<=n; i++){
    printf("%d\n", (int) (creal(A[i])+0.5));
    printf("%d\n", (int) (cimag(A[i])+0.5));
  }
  return 0;
}
