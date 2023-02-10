#ifndef DPOLY_H
#define DPOLY_H

/**
 * evaluates a polynomial of degree n with coefficients a[i] (i = 0...n)
 * at x and returns the result.
 *
 * res = a[n] * x^n + a[n-1] * x^(n-1) + ... a[1] * x + a[0]
 *
 * @param n degree of polynomial
 * @param a coefficients a[0] ... a[n]
 * @param x argument, at which the polynomial shall be evaluated
 * @param dx order of derivation
 * 
 * @return value of the n-th-degree polynomial with coefficients a[i] at x
 */
double dpoly(unsigned n, double * a, double x, unsigned dx);


#endif /* DPOLY_H */