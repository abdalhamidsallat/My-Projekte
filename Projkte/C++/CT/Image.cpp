
#include "Image.h"

Image::Image() {
	// TODO Auto-generated constructor stub

}

Image::~Image() {
	// TODO Auto-generated destructor stub
}

double Image::getVoxelSizeX(void) {
	int matSizeX = m_Pixel.getNCols();
	return ( matSizeX>0 ? m_FoVx/matSizeX : 0 );
}

double Image::getVoxelSizeY(void) {
	int matSizeY = m_Pixel.getNRows();
	return ( matSizeY>0 ? m_FoVy/matSizeY : 0 );
}

void Image::resetImageSize(int ImgSizeX, int ImgSizeY) {
	m_Pixel.reset(ImgSizeY,ImgSizeX);
}

void Image::set(int x, int y, double value) {
	m_Pixel.set(y,x,value);
}
