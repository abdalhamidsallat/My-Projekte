#include <iostream>
#include <math.h>


#include "Image.h"
#include "Sinogram.h"
#include "matrix.h"

#define pi 3.14159265359
using namespace std;

int main() {
	
Sinogram Data_Sinogram;
Image  img_Rekonstruktion;

string  Data_Sinogram_text = "C:/Users/49157/Desktop/Median/Median/Sinogram_InVivo.txt";

Data_Sinogram.loadSinogramFromTextFile(Data_Sinogram_text);

int Phi = Data_Sinogram.getNumberOfAngularPositions();
printf("%i",Phi);
Data_Sinogram.setLengthOfDetectorArray(750);

double dek_leng= Data_Sinogram.getLengthOfDetectorArray();

img_Rekonstruktion.resetImageSize(350,400);

int X_Columns= img_Rekonstruktion.getImageSizeX();

int Y_Rows= img_Rekonstruktion.getImageSizeY();
printf("%i %i", X_Columns, Y_Rows);

img_Rekonstruktion.setFoVx(450);
img_Rekonstruktion.setFoVy(400);
double dx = img_Rekonstruktion.getVoxelSizeX();
double dy = img_Rekonstruktion.getVoxelSizeY();


double Fov_X_mm = img_Rekonstruktion.getFoVx();
double Fov_Y_mm = img_Rekonstruktion.getFoVy();
printf("\n%f %f", Fov_X_mm, Fov_Y_mm);

float Dek_Bereite= 750/557;
int Dek_anz= 557;

/*int Bild_Matrix[X_Columns][Y_Rows];
	for(int i= 0 ; i< X_Columns ; i++)
		for(int j = 0 ; j<Y_Rows ; j++)
			Bild_Matrix[i][j]=0;
*/

double rx , ry , Lateralindex , Lateralposition;
	 // double Grauwert [180];
		/*double a [180];
		for(int i= 0 ; i< 180 ; i++)
			a[i] = 0;
*/

for (double i = 1; i < 350; i++)
{
	for (double j = 1; j <400; j++)
	{
		rx = ((-Fov_X_mm / 2) + ((i - 1) * dx));
		ry = ((Fov_Y_mm / 2) - ((j - 1) * dy));
		double a = 0.0;
		for (double winkel = 0; winkel < Phi; winkel++) {

			Lateralposition =( rx * (cos(winkel * pi / 180))) + (ry * (sin(winkel * pi / 180)));
			//Lateralindex = round(557 / 2) + round(Lateralposition / Dek_Bereite);

			 a = Data_Sinogram(Lateralposition, winkel) + a;
		


		}
	img_Rekonstruktion.set(i, j, a);
	}

}
string  output = "C:/Users/49157/Desktop/Median/Median/output.txt";
img_Rekonstruktion.storeImageInTextFile(output);
	return 0;
}

