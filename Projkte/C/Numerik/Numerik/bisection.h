#ifndef BISECTION_H
#define BISECTION_H

/**
 * evaluates a minimum of a polynomial
 * by using the golden section methode
 *
 * res = x_min
 *
 * @param n degree of polynomial
 * @param x0 start argument of the initalization
 * @param a coefficients a[0] ... a[n] * 
 * @param eta initial step size
 * @param eps accuracy of the minimum
 *
 * @return value of the minimum (x-coordinate)
 */
double bisection(unsigned n, double * a, double x0, double x1, double eps);


#endif /* BISECTION_H */