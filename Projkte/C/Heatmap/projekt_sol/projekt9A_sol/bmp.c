#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "bmp.h"


/**
 * calculates the number of bytes
 * is used to write a bmp-file
 *
 *
 * @param bmp pointer of bitmap-file
 * @param byte byte
 * @param N numbers
 *
 */
static void bmp_write_N_byte(FILE * bmp, uint64_t byte, int N);



void bmp_create(const char * bmp_name, const uint32_t * data, int width, int height)
{
    if (data == NULL || width <= 0 || height <= 0) {
        fprintf(stderr, "Error: invalid parameters for bmp-file \n");
        exit(EXIT_FAILURE);
    }

    FILE * fpbmp = fopen(bmp_name, "wb");
    if (fpbmp == NULL) {
        fprintf(stderr, "Error: cannot open bmp-file \n");
        exit(EXIT_FAILURE);
    }
    /* BMP header */
    // magic code: BM
    fputc('B', fpbmp);
    fputc('M', fpbmp);
    // (4 Byte, uint32_t) size of BMP file in byte
    // = bmp_file_header + bmp_info_header + pixels
    bmp_write_N_byte(fpbmp, 14 + 40 + width*height*4,4);
    // (4 Byte, uint32_t) reserved, default: 0
    bmp_write_N_byte(fpbmp, 0, 4);
    bmp_write_N_byte(fpbmp, 54, 4);

    /* BMP info */
    //(4 Byte, uint32_t) size of bmp_info in byte
    bmp_write_N_byte(fpbmp, 40, 4);
    //(4Byte, int32_t) width of bitmap in #pixels
    bmp_write_N_byte(fpbmp, width, 4);
    // (4 Byte, int32_t) height of bitmap in #pixels. negative: top-bottom
    bmp_write_N_byte(fpbmp, -height, 4);
    // (2 Byte, uint16_t) unused (number of color layers)
    bmp_write_N_byte(fpbmp, 1, 2);
    // (2 Byte, uint16_t) color format in bpp; must be one of
    // 1, 4, 8, 16, 24 or 32, where colors are indexed for 1, 4 and 8 bpp
    bmp_write_N_byte(fpbmp, 32, 2);
    // (4 Byte, uint32_t) compression: 0 = none
    bmp_write_N_byte(fpbmp, 0, 4);
    // (4 Byte, uint32_t) size for all pixels (including padding) in byte
    bmp_write_N_byte(fpbmp, width * height * 4, 4);
    // (4 Byte, int32_t) horizontal resolution, typically 0 for BMP
    bmp_write_N_byte(fpbmp, 0, 4);
    // (4 Byte, int32_t) vertical resolution, typically 0 for BMP
    bmp_write_N_byte(fpbmp, 0, 4);
    // (4 Byte, uint32_t) number of entries in color table: 0 = none
    bmp_write_N_byte(fpbmp, 0, 4);
    // (4 Byte, uint32_t) number of important colors; 0 = all are important
    bmp_write_N_byte(fpbmp, 0, 4);

    /* pixels */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            bmp_write_N_byte(fpbmp, data[y * width + x], 4);
        }
    }

    fclose(fpbmp);
}



void bmp_write_N_byte(FILE * bmp, uint64_t byte, int N)
{
    for(int i = 0; ((i<N) && (i < 8)); i++) {
        fputc(byte % 256, bmp);
        byte /= 256;
    }
}
