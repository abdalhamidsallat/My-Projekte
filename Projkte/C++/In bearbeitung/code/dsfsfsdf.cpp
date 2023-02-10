#include "Mesh_Einlesen.h"
#include "Quelle.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

using namespace std;

int main() {

	vector<vector<double>>Knoten;
	vector<vector<int>>Element;
	string Name = ("k.vtk");
	Mesh_Einlesen(Knoten, Element, Name);
	double q2;
	double sq[8];
	 q(Knoten, Element, q2,sq);


	for (int i = 0; i < Element.size(); i++)
	{
		for (int j = 0; j < Element[i].size(); j++)
		{
			//cout << Element[i][j] << "\t";
		}//cout << endl;


	}

	cout << q2;
	for (int  i = 0; i < 8; i++)
	{
		cout << sq[i]<< endl;
	}
	











	return 0;
}