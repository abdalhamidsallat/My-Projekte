#include "Mesh_Einlesen.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

using namespace std;

void q (vector<vector<double>>&Knoten,vector<vector<int>>&Element,double &q2,double *sq) {



	vector<vector<double>>A_Vector(8);
	vector<vector<double>>B_Vector(8);
	vector<vector<double>>C_Vector(8);


	int index = Element.size() - 10;
	double P0 = Knoten[Element[index][1]][0] - Knoten[Element[index][0]][0];
	double a0 = P0 / sqrt(pow(P0, 2));
	double b1 = -a0;
	A_Vector[0].push_back(a0);
	B_Vector[1].push_back(b1);

	double P1 = Knoten[Element[index][3]][1] - Knoten[Element[index][0]][1];
	double b0 = P1 / sqrt(pow(P1, 2));
	double a3 = -b0;
	A_Vector[3].push_back(a3);
	B_Vector[0].push_back(b0);

	double P2 = Knoten[Element[index][4]][2] - Knoten[Element[index][0]][2];
	double c0 = P2 / sqrt(pow(P2, 2));
	double c4 = -c0;
	C_Vector[0].push_back(c0);
	C_Vector[4].push_back(c4);

	double P3 = Knoten[Element[index][2]][1] - Knoten[Element[index][1]][1];
	double a1 = P3 / sqrt(pow(P3, 2));
	double b2 = -a1;
	A_Vector[1].push_back(a1);
	B_Vector[2].push_back(b2);

	double P4 = Knoten[Element[index][5]][2] - Knoten[Element[index][1]][2];
	double c1 = P4 / sqrt(pow(P4, 2));
	double c5 = -c1;
	C_Vector[1].push_back(c1);
	C_Vector[5].push_back(c5);

	double P5 = Knoten[Element[index][3]][0] - Knoten[Element[index][2]][0];
	double a2 = P5 / sqrt(pow(P5, 2));
	double b3 = -a2;
	A_Vector[2].push_back(a2);
	B_Vector[3].push_back(b3);

	double P6 = Knoten[Element[index][6]][2] - Knoten[Element[index][2]][2];
	double c2 = P6 / sqrt(pow(P6, 2));
	double c6 = -c2;
	C_Vector[2].push_back(c2);
	C_Vector[6].push_back(c6);

	double P7 = Knoten[Element[index][7]][2] - Knoten[Element[index][3]][2];
	double c3 = P7 / sqrt(pow(P7, 2));
	double c7 = -c3;
	C_Vector[3].push_back(c3);
	C_Vector[7].push_back(c7);

	double P8 = Knoten[Element[index][5]][0] - Knoten[Element[index][4]][0];
	double b4 = P8 / sqrt(pow(P8, 2));
	double a5 = -b4;
	A_Vector[5].push_back(a5);
	B_Vector[4].push_back(b4);

	double P9 = Knoten[Element[index][7]][1] - Knoten[Element[index][4]][1];
	double a4 = P9 / sqrt(pow(P9, 2));
	double b7 = -a4;
	A_Vector[4].push_back(a4);
	B_Vector[7].push_back(b7);

	double P10 = Knoten[Element[index][6]][1] - Knoten[Element[index][5]][1];
	double b5 = P10 / sqrt(pow(P10, 2));
	double a6 = -b5;
	A_Vector[6].push_back(a6);
	B_Vector[5].push_back(b5);

	double P11 = Knoten[Element[index][7]][0] - Knoten[Element[index][6]][0];
	double b6 = P11 / sqrt(pow(P11, 2));
	double a7 = -b6;
	A_Vector[7].push_back(a7);
	B_Vector[6].push_back(b6);



	for (int i = 0; i < A_Vector.size(); i++)
	{
		 sq[i] = A_Vector[i][0] * B_Vector[i][0] * C_Vector[i][0];
		cout << sq[i] << endl;
	}
		// return q2;
}