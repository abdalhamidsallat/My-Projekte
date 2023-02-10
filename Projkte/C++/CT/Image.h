#ifndef IMAGE_H_
#define IMAGE_H_

#include "matrix.h"

using namespace std;

class Image
{

	matrix m_Pixel;
	double m_FoVx;      // Field-of-view in [mm] in x-direction
	double m_FoVy;	    // Field-of-view in [mm] in y-direction


public:

	// Constructor and desctructor
	Image();
	virtual ~Image();

	// Assginment operator: img(x,y) gives access to the element in pixel location (x,y)
	double& operator() (int x, int y) { return m_Pixel(y,x); };

	// Load the image specified by "fName". The image has to be stored as a space separated ASCII file
	bool loadImageFromTextFile(string fName) { return m_Pixel.loadFromFile(fName); };

	// Save the image specified by "fName". The image will  be stored as a space separated ASCII file
	bool storeImageInTextFile(string fName) { return m_Pixel.storeMatrixInTextFile(fName); };

	// Print the image grey values
	void print(void) { m_Pixel.print(); };

	// Return the number of pixels in x-direction
	int getImageSizeX(void) { return m_Pixel.getNCols(); };

	// Return the number of pixels in y-direction
	int getImageSizeY(void) { return m_Pixel.getNRows(); };

	// Return the Field-of-view in x-direction in [mm]
	double getFoVx(void) {return m_FoVx; };

	// Return the Field-of-view in y-direction in [mm]
	double getFoVy(void) {return m_FoVy; };

	// Return the voxel size in x-direction in [mm]
	double getVoxelSizeX(void);

	// Return the voxel size in y-direction in [mm]
	double getVoxelSizeY(void);

	// Set the Field-of-view in x-direction to "FoVx" (in [mm])
	void setFoVx(double FoVx) {m_FoVx = FoVx;};

	// Set the Field-of-view in y-direction to "FoVy" (in [mm])
	void setFoVy(double FoVy) {m_FoVy = FoVy;};

	// Reset the image to a size of "ImgSizeX" x "ImgSizeY" (in pixels) and fill pixels with zeros
	void resetImageSize(int ImgSizeX, int ImgSizeY);

	// Set the pixel at location ("x","y") to "value".
	void set(int x, int y, double value);
};

#endif /* IMAGE_H_ */
