#include <iostream>
#include <fstream>
#include<vector>


using namespace std;


int main() {


	int Qu = 1;   // Würfel in euner Richtung
	

	double d = 0.1; // Abstand zwischen Koordinaten in einer Ebene

	int Knoten_Zahl = ((Qu + 1) * (Qu + 1) * (Qu + 1)); // Anzahl der Knoten
	int Elementen_Zahl = (Qu * Qu * Qu);                // Anzahl der Elemente || Anzahl der gesammten Würfel
	
	vector < vector<double>> Knoten_Liste(Knoten_Zahl, vector<double>(3));

	int ind = 0;

	for (int j = 0; j <= Qu; j++) {
		double z = j * d;

		for (int i = 0; i <= Qu; i++)
		{
			double y = i * d;
			for (int k = 0; k <= Qu; k++)
			{
				double x = k * d;
				Knoten_Liste[ind][0] = x;
				Knoten_Liste[ind][1] = y;
				Knoten_Liste[ind][2] = z;
				ind++;
			}

		}
	}
	// Das braucht man für die Nummerierung der Knoten und die Bildung von Elementen
	vector<int > Knoten_Nummer_Liste(Knoten_Zahl, 0);
	int Knotennummer = 0;

	for (int j = 0; j < Knoten_Zahl; j++) {

		Knoten_Nummer_Liste[j] = Knotennummer;

		//cout << " " << Knoten_Nummer_Liste[j];
		Knotennummer++;

		cout << endl;
	}


	vector < vector<int>> Element_Knoten_Liste(Elementen_Zahl, vector<int>(9));

	int a = 0;
	int t = 0;
	int p = 2;
	int r = 1;
	for (int s = 0; s < Qu; s++) // Qu ist auch Anzahl der Schichten (eine Schicht sind die Elemente in einer Ebene) 
	{
		for (int j = 0; j < (Qu * Qu); j++) // (Qu*Qu) Anzahl der Elemente in einer Schicht
		{
			if ((a + 1) % (Qu + 1) == 0) // am Ende jeder Zeile werden wir der letzte Knoten nicht betrachten
			{
				a++;
			}
			if (a == (((Qu + 1) * (Qu + 1) - 1) - Qu)) // auch die obere zeile  wird nicht betrachtet
			{


				a = a + Qu + 1;                      //  Qu+1 addieren, um die nächste Schicht zu erreichen

			}

			if (a == ((((Qu + 1) * (Qu + 1) - 1) - Qu) * (p)) + ((Qu + 1) * r)) // die obere Zeile in allen Schichten werden nicht betrachtet
			{
				a = a + Qu + 1;
				p++;
				r++;
			}
			Element_Knoten_Liste[t][0] = 8;


			Element_Knoten_Liste[t][1] = Knoten_Nummer_Liste[a];
			Element_Knoten_Liste[t][2] = Knoten_Nummer_Liste[a + 1];
			Element_Knoten_Liste[t][4] = Knoten_Nummer_Liste[(a + 1) + (Qu - 1) + 1];
			Element_Knoten_Liste[t][3] = Knoten_Nummer_Liste[(a + 1) + (Qu - 1) + 2];
			Element_Knoten_Liste[t][5] = Knoten_Nummer_Liste[a + ((Qu + 1) * (Qu + 1))];
			Element_Knoten_Liste[t][6] = Knoten_Nummer_Liste[a + ((Qu + 1) * (Qu + 1)) + 1];
			Element_Knoten_Liste[t][8] = Knoten_Nummer_Liste[a + ((Qu + 1) * (Qu + 1)) + 1 + (Qu - 1) + 1];
			Element_Knoten_Liste[t][7] = Knoten_Nummer_Liste[a + ((Qu + 1) * (Qu + 1)) + 1 + (Qu - 1) + 2];



			a++;
			t++;
		}

	}







	// VTK-Datei erstellen

	std::ofstream file;
	file.open("50x50x50.vtk");
	// Der erste Teil ist die Dateiversion und der Bezeichner
	file << "# vtk DataFile Version 4.0" << endl; 
	// Header Würfel // Format der Datei "ASCII" // die Datensatzstruktur
	file << "3D Würfel" << endl << "ASCII" << endl << "DATASET UNSTRUCTURED_GRID" << endl;
	// die Datensatzattribute 
	file << "POINTS " << Knoten_Zahl << " double" << endl;
	for (int i = 0; i < Knoten_Liste.size(); i++)
	{

		for (int j = 0; j < Knoten_Liste[i].size(); j++)
		{
			file << Knoten_Liste[i][j] << " ";

		}
		file << endl;

	}
	// 9 bedeutet, dass jedes Element 8 Knoten hat und am Anfang kommt die Nummer 8 mit der Bedeutung, dass das Element 8 Knoten hat insgesamt (9) Zahlen
	// Und danach die Summe von alle Zahlen, ist die Anzahl von der gesamten Würfel * 9
	// Cells erklärt der Anzahl von der gesamten Würfel

	file << "CELLS " << Elementen_Zahl << " " << 9 * Elementen_Zahl << endl;
	for (int i = 0; i < Element_Knoten_Liste.size(); i++)
	{

		for (int j = 0; j < Element_Knoten_Liste[i].size(); j++)
		{
			file << Element_Knoten_Liste[i][j] << " ";

		}
		file << endl;

	}
	// Dieser Teil erklärt, mit welchem Typ von VTK_HEXAHEDRON(=12)
	// 12 muss sowiel geschrieben werden, wie die Anzahl von der gesamten Würfel
	file << "CELL_TYPES " << Elementen_Zahl << endl;
	for (int j = 0; j < Elementen_Zahl; j++)
	{
		file << 12 << endl;

	}







	file.close();

}
