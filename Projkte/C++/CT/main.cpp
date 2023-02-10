#include"Image.h"
#include"Sinogram.h"
#include <iostream>
#include "matrix.h"
#define pi 3.14159265359
using namespace std;

int main()
{

Sinogram Median_sino;
Image  Mohammed_img;


string Median_sino= "Sinogram_InVivo.txt";

Median_sino.loadSinogramFromTextFile(Median_sino);

int Phi= Median_sino.getNumberOfAngularPositions();

Median_sino.setLengthOfDetectorArray(750);

double dek_leng= Median_sino.getLengthOfDetectorArray();


int X_Columns= Mohammed_img.getImageSizeX();

int Y_Rows= Mohammed_img.getImageSizeY();
Mohammed_img.resetImageSize(350,400);

double dx = Mohammed_img.getVoxelSizeX();
double dy = Mohammed_img.getVoxelSizeY();

Mohammed_img.setFoVx(450);
Mohammed_img.setFoVy(400);
double Fov_X_mm = Mohammed_img.getFoVx();
double Fov_Y_mm = Mohammed_img.getFoVy();

float Dek_Bereite= 750/557;
int Dek_anz= 557;

int Bild_Matrix[X_Columns][Y_Rows];
	for(int i= 0 ; i< X_Columns ; i++)
		for(int j = 0 ; j<Y_Rows ; j++)
			Bild_Matrix[i][j]=0;


double rx , ry , Lateralindex , Lateralposition;
		double Grauwert [180];
		double a [180];
		for(int i= 0 ; i< 180 ; i++)
			a[i] = 0;

for(int i = 0 ; i<X_Columns ; i++)
		for(int j = 0 ; j<Y_Rows ; j++)
		{
			rx= (-Fov_X_mm/2+((i-1)*dx));
			ry= (Fov_Y_mm/2-((j-1)*dy));

			for(int phi = 0 ; phi<= 180 ; i++){

			Lateralposition= rx*cos((phi*pi/180))+ry*sin((phi*pi/180));
				Lateralindex= round(557/2)+round( Lateralposition/Dek_Bereite);

				Grauwert[phi] = Sinogram(Lateralindex , phi);

				a += Grauwert[phi];

			}
			Bild_Matrix[i][j]=a;
		}
		Mohammed_img.storeImageInTextFile(),
		return 0;
}

