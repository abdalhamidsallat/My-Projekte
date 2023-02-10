#include <math.h>
#include "Sinogram.h"
#include <iostream>

using namespace std;

Sinogram::Sinogram()
{
	// TODO Auto-generated constructor stub

}

Sinogram::~Sinogram()
{
	// TODO Auto-generated destructor stub
}


double Sinogram::operator() (double p, double phi)
{
	int angularIdx;
	int latPosIdx;
	double retVal=-1000;

	// determine correct (nearest neighbor) index for the current pair (p,phi)
	angularIdx = round(phi * (getNumberOfAngularPositions()/180 ))+1;
	latPosIdx = round( (m_pMax + p)/getSizeOfSingleDetectorElement());

	// Check size
	//cout << angularIdx <<"  "<<latPosIdx <<endl;
	if( angularIdx < 1 || angularIdx > getNumberOfAngularPositions() ||
		latPosIdx < 1 || latPosIdx > getNumberOfDetectorElements()	) {
	}
	else {
		retVal = m_RawData(latPosIdx, angularIdx);
	}

	return retVal;
}


double Sinogram::getSizeOfSingleDetectorElement(void)
{
	int numDetectorElements = getNumberOfDetectorElements();
	return ( numDetectorElements>0 ? getLengthOfDetectorArray()/numDetectorElements : 0 );
}

