#include <stdio.h>
#include "dpoly2.h"

signed int typextrem (unsigned n, double * a, double x_ext) {
    double expt;
    double y_ext = dpoly2(n, a, x_ext);
    
    printf("Art des Extremwertes: ");
    if (y_ext < 0) {
        expt = -1;
        printf("Minimum.\n");
    } else if (y_ext > 0) {
        expt = 1;
        printf("Maximum.\n");
    } else {
        expt = 0;
        printf("nicht eindeutig.");
    }

    return expt;
    
}