#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
using namespace std;

void Mesh_Einlesen(vector<vector<double>>& Knoten,
	vector<vector<int>>& Elemente, string Name)
{


	int Knoten_Zahl, Elementen_Zahl, Z,A;
	ifstream Mesh_L;
	Mesh_L.open(Name);
	
	string Zeile,a,C,d;
	for(int i=0;i<4;i++)
	{
		getline(Mesh_L, Zeile);
		//cout << Zeile << endl;
	}
	
	
	Mesh_L >>a >> Knoten_Zahl >> d;
	//cout << Knoten_Zahl<<endl;
	//cout << d << endl;
	// Anpassen der Grüsse von angegbenen Vektoren
     Knoten.resize(Knoten_Zahl, vector<double>(3));
	


	// Knoten Einlesen
	for (int i = 0; i < Knoten_Zahl; i++) 
	{

		for (int j = 0; j < 3; j++)
		{
			Mesh_L >> Knoten[i][j];
			//cout << Knoten[i][j];
		}cout << endl;
	}
	Mesh_L >> C >> Elementen_Zahl >>Z;
	//cout << Elementen_Zahl;
	Elemente.resize(Elementen_Zahl, vector<int>(0));
	//// Elementen Einlesen 
	int knotee;
	for (int i = 0; i < Elementen_Zahl; i++) {
		Mesh_L >> A;

		for (int j = 0; j < A; j++) {
			Mesh_L >> knotee;

			Elemente[i].push_back(knotee);

			//cout << Elemente[i][j] << "\t";

		}cout << endl;


	}

}
#pragma once
