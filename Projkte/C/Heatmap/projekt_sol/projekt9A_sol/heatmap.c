#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "bmp.h"

#define ARG_FILENAME   1
#define ARG_HEIGHT     2
#define ARG_WIDTH      3
#define NUM_ARGS       4

#define ARG_IPOL       4
#define NUM_ARGS_OPT   5


#define MIN(a,b)  ((a) < (b) ? (a) : (b))


// BMP colors
static const uint32_t BMP_COLORS[] = {
    0x00000066,  // dark blue
    0x000000FF,  // blue
    0x000080FF,  // light blue
    0x0000FF80,  // blue green
    0x0000CC00,  // green
    0x0080E020,  // light green
    0x00FFFF33,  // yellow
    0x00FF9900,  // light orange
    0x00FF6600,  // orange
    0x00CC0033   // red
};
#define BMP_BG_VALUE      0x00FFFFFF   // white
#define BMP_NUM_COLORS   (sizeof(BMP_COLORS)/sizeof(BMP_COLORS[0]))


// number of clusters
#define CLUSTER_NUM 10


enum {
    IPOL_RASTER,
    IPOL_NEAREST,
    IPOL_LINEAR,
    IPOL_NUM
};


typedef struct {
    double x;
    double y;
    double z;
} seafloor_t;


// FIXME this lengthy main is very unpretty!

int main(int argn, char * argv[])
{
    // read from command line input
    if (argn < NUM_ARGS || argn > NUM_ARGS_OPT) {
        fprintf(stderr, "\n USAGE: %s FILENAME HEIGHT WIDTH [IPOL]\n\n", argv[0]);
        fprintf(stderr, "  Draw a Bitmap with side height HEIGHT and side width WIDTH from FILENAME (CSV file)\n"
            "  Optional parameter IPOL determines interpolation method:\n"
            "    - 0 rasterization\n"
            "    - 1 nearest neighbor\n"
            "    - 2 linear interpolation\n"
        );
        exit(1);
    }

    const char * fn = argv[ARG_FILENAME];
    unsigned bmpHeight = atoi(argv[ARG_HEIGHT]);
    unsigned bmpWidth  = atoi(argv[ARG_WIDTH]);
    //printf("%u x %u\n", bmpHeight, bmpWidth);

    // check if these are reasonable and positive, in particular
    if (bmpHeight > 65536 || bmpWidth > 65536) {
        fprintf(stderr, "ERROR: out of Range. Check your input WIDTH %u or HEIGHT %u\n", bmpWidth, bmpHeight);
        exit(1);
    }

    int ipol;
    if (argn > ARG_IPOL) {
        ipol = atoi(argv[ARG_IPOL]);
        if (ipol < 0 || ipol >= IPOL_NUM) {
            fprintf(stderr, " draw a Bitmap with side height HEIGHT and side width WIDTH from FILENAME (CSV file)\n\n");
            exit(1);
        }
    }

    // input file
    FILE * fpcsv;

    // open file
    fpcsv = fopen(fn, "r");
    if (fpcsv == NULL) {
        fprintf(stderr, "ERROR: could not open CSV file '%s'\n\n", fn);
        exit(1);
    }

    // count lines (entries)
    // NOTE we assume all valid lines
    seafloor_t lim[2] = { {DBL_MAX, DBL_MAX, DBL_MAX}, {DBL_MIN, DBL_MIN, DBL_MIN} };
    double dx, dy;
    fscanf(fpcsv,"%lf,%lf,%lf", &lim[0].x, &lim[1].x, &dx);
    fscanf(fpcsv,"%lf,%lf,%lf", &lim[0].y, &lim[1].y, &dy);

    // number of x/y values
    unsigned numX = 1 + round(lim[1].x - lim[0].x) / dx;
    unsigned numY = 1 + round(lim[1].y - lim[0].y) / dy;
    printf("file has %u x- and %u y-values\n", numX, numY);

    // check this
    if (bmpWidth < numX || bmpHeight < numY) {
        fprintf(stderr, "ERROR: size of BMP (in pixels) is lower than input data span\n\n");
        exit(1);
    }

    // get memory for data from file
    seafloor_t * inData;
    inData = (seafloor_t *) malloc(numX * numY * sizeof(seafloor_t));
    if (inData == NULL) {
        fprintf(stderr, "ERROR: not enough memory\n\n");
        exit(1);
    }

    // read data and find min/max z-values on the fly
    for (unsigned i = 0; i <  numX * numY; i++) {
        if (fscanf(fpcsv,"%lf,%lf,%lf", &inData[i].x, &inData[i].y, &inData[i].z) == EOF) {
            fprintf(stderr, "ERROR: premature EOF (expected %u lines but found %u)\n\n", numX*numY, i+1);
            exit(1);
        }

        // check that x, y in bounds!
        if (inData[i].x < lim[0].x || inData[i].x > lim[1].x) {
            fprintf(stderr, "ERROR: Input data x out of bounds.\n\n");
            exit(1);
        }
        
        if (inData[i].y < lim[0].y || inData[i].y > lim[1].y) {
            fprintf(stderr, "ERROR: Input data y out of bounds. \n\n");
            exit(1);
        }

        if (inData[i].z < lim[0].z){
            lim[0].z = inData[i].z;
        } else if (inData[i].z > lim[1].z) {
            lim[1].z = inData[i].z;
        }
    }

    // done with file I/O
    fclose(fpcsv);

    /* create a bitmap and make it all white */
    uint32_t * outData = (uint32_t*) malloc(bmpWidth * bmpHeight * sizeof(uint32_t));
    if (outData == NULL) {
        fprintf(stderr, "ERROR: not enough memory\n\n");
        exit(1);
    }

    // fill white to better spot issues and as bg for rasterization
    for (long i = 0; i < bmpWidth * bmpHeight; i++) {
        outData[i] = BMP_BG_VALUE;
    }

    // get deltas in x,y direction and z-clusters
    seafloor_t inDelta;
    inDelta.x = lim[1].x - lim[0].x;
    inDelta.y = lim[1].y - lim[0].y;
    inDelta.z = lim[1].z - lim[0].z;
    //printf("lim: %lf %lf %lf\n", inDelta.x, inDelta.y, inDelta.z);

    // ToDo use function for separate methodes
    double z = 0.0;
    double x, y;
    for (unsigned i = 0; i < numX; i++) {
        for ( unsigned j = 0; j < numY; j++) {
            x = inData[i*numY + j].x;
            y = inData[i*numY + j].y;
            z = inData[i*numY + j].z;
            unsigned row = MIN((y-lim[0].y) * bmpHeight / (inDelta.y), bmpHeight - 1);
            unsigned col = MIN((x-lim[0].x) * bmpWidth / (inDelta.x),bmpWidth-1);

            if (ipol == IPOL_RASTER) {            
                uint8_t pixIdx = (z - lim[0].z) / inDelta.z * CLUSTER_NUM;
                outData[row * bmpWidth + col] = BMP_COLORS[pixIdx];
            } else {
                double xNext = inData[(i+1)*numY + j+1].x;
                double yNext = inData[(i+1)*numY + j+1].y;
                double zNext = inData[(i+1)*numY + j+1].z;
                unsigned rowNext = MIN((yNext-lim[0].y) * bmpHeight / (inDelta.y), bmpHeight - 1);
                unsigned colNext = MIN((xNext-lim[0].x) * bmpWidth / (inDelta.x),bmpWidth-1);

                unsigned datCol;
                unsigned datRow;
                uint8_t pixIdx;

                if (ipol == IPOL_NEAREST) {
                    for (datRow = row; datRow <= rowNext; datRow++) {

                        for (datCol = col; datCol <= colNext; datCol++) {
                            if (datRow > 0.5 * (row - rowNext) && datRow + 1 < numY) {
                            z = inData[(i+1)*numX + j+1].z;
                            }
                            if (datCol > 0.5 * (col - colNext) && datCol + 1 < numX) {
                            z = inData[(i+1)*numX + j+1].z;
                            } 
                            pixIdx = (z - lim[0].z) / inDelta.z * CLUSTER_NUM;
                            pixIdx = MIN(CLUSTER_NUM-1, pixIdx);  // stay in bounds
                            outData[datRow * bmpWidth + datCol] = BMP_COLORS[pixIdx];                                
                        }
                    }
                } else if (ipol == IPOL_LINEAR) { 

                                                          
                    double datZ;
                    double datX;
                    double datY;
                    for (datCol = col; datCol <= colNext; datCol ++) {
                        //zNext = inData[datCol * numY + j].z;
                        for (datRow = row; datRow <= rowNext; datRow++) {                            
                            //x actual -> interpol between x and xNext from input data
                            datX = lim[0].x + ((double) datCol * (inDelta.x)) / bmpWidth;
                            printf("x: %lf; xNext; %lf; datX %lf \n",x, xNext, datX);

                            //y actual -> interpol between y and yNext from input data
                            datY = lim[0].y + ((double) (datRow) * (inDelta.y)) / bmpHeight;
                            printf("y: %lf; yNext; %lf; datY %lf \n",y, yNext, datY);

                            // ipol z in y direction

                            datZ = z + (zNext - z)/(yNext - y) * (datY - y);                
                                                        
                            printf("z: %lf; zNext; %lf; datZ: %lf\n",z, zNext, datZ);

                            pixIdx = (datZ - lim[0].z) / inDelta.z * CLUSTER_NUM;
                            pixIdx = MIN(CLUSTER_NUM-1, pixIdx);  // stay in bounds
                            outData[datRow * bmpWidth + datCol] = BMP_COLORS[pixIdx];
                        }
                        double zDirX = inData[(i+1) * numY +j].z;
                        z = z + (zDirX - z) / (xNext - x) * (datX - x);
                    }
                }

            } 
        }
        
        
    }


    // .bmp-file schreiben
    bmp_create("heatmap.bmp", outData, bmpWidth, bmpHeight);

    // and clean up
    free(inData);
    free(outData);

    return 0;
}
