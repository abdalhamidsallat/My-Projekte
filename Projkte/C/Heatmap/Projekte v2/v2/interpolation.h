#ifndef INTERPOLATION_H
#define INTERPOLATION_H
/**
 * interpolates data points in a equal sized or bigger map
 * @param data_points array of z values
 * @param map map data_points is interpolated to
 * @param WIDTH_DATA width of data_points array
 * @param HEIGHT_DATA height of data_points array
 * @param WIDTH_MAP width of map array
 * @param HEIGHT_MAP heifght of map array
*/

int interpolieren(double *data_points, double *map, unsigned WIDTH_DATA, unsigned HEIGHT_DATA, unsigned WIDTH_MAP, unsigned HEIGHT_MAP);


#endif
