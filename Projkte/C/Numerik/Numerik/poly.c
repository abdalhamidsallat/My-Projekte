double poly(unsigned n, double * a, double x)
{
  double res = a[0];
  double xn = x;
  //printf("Koeffizient: %g \n", a[0]);
  for (unsigned i = 1; i <= n; i++) {
    res += a[i] * xn;
    xn *= x;
    //printf("Koeffizient: %g \n", a[i]);
  }
  return res;
}