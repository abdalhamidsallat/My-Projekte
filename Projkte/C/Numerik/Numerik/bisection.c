#include <stdio.h>
#include "dpoly1.h"

double bisection (unsigned n, double * a, double x0, double x1, double eps) {
    
    //Initialisierung
    double absy;
    double y0;
    double y1;
        
    //Intervall verkleinern
    do {
        y0 = dpoly1(n, a, x0);
        y1 = dpoly1(n, a, x1);

        if (y0 < y1){
            absy = y1-y0;
        } else {
            absy = y0-y1;
        }

        double xm = (x0+x1) / 2.0;
        double ym = dpoly1(n, a, xm);

        if (y0*ym < 0){
            x1 = xm;
        } else if (y1*ym < 0){
            x0 = xm;
        }
    } while (absy > eps); //Abbruchkriterium

    double x_ext = (x1+x0)/2.0;
    return x_ext;
}
