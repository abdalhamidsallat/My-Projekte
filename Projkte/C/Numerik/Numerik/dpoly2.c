double dpoly2(unsigned n, double * a, double x) {
    double res = 2*a[2];
    double xn = x;
    //printf("Koeffizient: %g %u %g\n", xn,n, a[0]);
    for (unsigned i = 3; i<=n; i++) {
        res += a[i]*i*(i-1)*xn;
        xn *= x;
    }

    return res;
}