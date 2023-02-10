double dpoly1(unsigned n, double * a, double x) {
    double res = a[1];
    double xn = x;
    //printf("Koeffizient: %g %u %g\n", xn,n, a[0]);
    for (unsigned i = 2; i<=n; i++) {
        res += a[i]*i*xn;
        xn *= x;
    }

    return res;
}