#ifndef COLORS_H
#define COLORS_H
/**
 * transfers value of data points of map in color values in img
 * 
 * @param map array of data points
 * @param img array with colorcodes in bpp32
 * @param width width of map
 * @param height height of map
 * @param min minimum z value in map
 * @param max maximum z value in map
*/

void color_value(double * map, uint32_t * img, int width, int height, double min, double max); //weist z-Werten aus map[] Farbwerte in img[] zu

#endif