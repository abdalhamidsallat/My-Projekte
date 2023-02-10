#ifndef DPOLY1_H
#define DPOLY1_H

/**
 * evaluates a polynomial of degree n with coefficients a[i] (i = 0...n)
 * at x and returns the result.
 *
 * res = a[n] * x^n + a[n-1] * x^(n-1) + ... a[1] * x + a[0]
 *
 * @param n degree of polynomial
 * @param a coefficients a[0] ... a[n]
 * @param x argument, at which the polynomial shall be evaluated
 *
 * @return value of the n-th-degree polynomial with coefficients a[i] at x
 */
double dpoly1(unsigned n, double * a, double x);


#endif /* DPOLY1_H */