#ifndef BMP_H
#define BMP_H

#include <stdint.h>

/**
 * writes a file in bmp-format
 * structured as follows: 1.bmp-file-head :: 2.bmp-file-info :: 3.pixel data
 * pay attention to the format of the pixel data array
 *
 * @param bmp_name name of .bmp-file
 * @param data array with pixel points -> use index data[Y * BMP_WIDTH + X]
 * @param width width of bitmap
 * @param height height of bitmaüp
 *
 */
void bmp_create(const char * bmp_name, const uint32_t * data, int width, int height);


#endif /* BMP_H */
