#include <stdio.h>
#include "poly.h"
#include "dpoly1.h"
#include "dpoly2.h"
#include "bisection.h"
#include "typextrem.h"

#define POLY_DEG 4
#define EPS 0.005

//Funktionen können auch hier programmiert werden
//Alternativ: Extra Dateien

int main(void)
{
    //Initialisierung
    double x0 = -1.25;
    double x1 = 3.0;
    double a[POLY_DEG+1] = { 2.0, -1/10.0,0.0, -1.0, 3.0/8.0};

    //Polynom-Funktion testen
    double y0 = poly(POLY_DEG, a, x0);
    printf("f(%g) = %g\n", x0, y0);

    double dy0 = dpoly1(POLY_DEG, a, x0);
    printf("f'(%g) = %g\n", x0, dy0);

    double ddy0 = dpoly2(POLY_DEG, a, x0);
    printf("f''(%g) = %g\n", x0, ddy0);
    
    //Extremwert mit Bisektionsverfahren
    double x_ext = bisection(POLY_DEG, a, x0, x1, EPS);
    double y_ext = poly(POLY_DEG, a, x_ext);

    printf("Extremwert bei f(%g) = %g\n", x_ext, y_ext);

    //Art des Extremwertes
    signed int spec = typextrem (POLY_DEG, a, x_ext);
    printf("Specify: %d \n", spec);

  return 0;
}