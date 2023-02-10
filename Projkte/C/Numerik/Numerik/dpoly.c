#include <stdio.h>

double dpoly(unsigned n, double * a, double x, unsigned dx) {
    double res;
    int coeff = 1;

    double xn = x;

    if (dx == 0){
        res = a[0];
        for (unsigned i = dx+1; i <=n; i++) {
            res += a[i] * xn;
            xn *= x;
        }
    } else {
        for (unsigned i = dx+1; i <= n; i++) {
            for (unsigned n = 0; n < dx; n++) {
                coeff *= (i-n);
                //printf("%d\n", coeff);
            }
            res += coeff * a[i] * xn;
            //printf("%lf\n", res);
            xn *= x;
            coeff = 1;
        }
    }

    return res;
}