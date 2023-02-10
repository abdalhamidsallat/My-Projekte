#ifndef TYPEXTREM_H
#define TYPEXTREM_H

/**
 * decision of the type of the extremum
 * minimum or maximum
 *
 * res = expt
 *
 * @param n degree of polynomial
 * @param a coefficients a[0] ... a[n] * 
 * @param x_ext argument of the extremum
 *
 * @return decision type of extremum (min or may)
 */
signed int typextrem (unsigned n, double * a, double x_ext);


#endif /* TYPEXTREM_H */