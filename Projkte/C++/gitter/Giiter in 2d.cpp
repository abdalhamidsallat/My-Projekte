#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace std;

int main()

{
    ofstream Knoten;

    cout << "Geben Sie die Anzahl von Quadraten auf eine Ebene"<< endl;     // Anzahl der Elemente auf eine Ebene
    int N;
    scanf_s("%i",&N);
        
    int nn = (N + 1) * (N + 1); //Anzahlvon den Knoten
    int ne = N * N; // Anzahl von den Elementen
    double H = 0.5;
    
    Knoten.open("Aufgabe00.OFF");

    Knoten << "OFF" << endl;
    Knoten << nn <<"\t"<< ne <<"\t"<<  0 << endl;

    vector<vector<double> >C(nn, vector<double>(3)); //Vector der Höhe nn und 3 lang
    int cnt = 0;
    for (int j = 0; j <= N; j++) {
        double y = H * j;
        for (int i = 0; i <= N; i++) {
            double x = H * i;
            C[cnt][0] = x; // x,y,z koordinaten für jede Knote
            C[cnt][1] = y;
            C[cnt][2] = 0.0;
            cnt = cnt + 1;
        }
    }
    
    //Elemente mit deren Knoten 
    int e = 0;
    vector<vector<int> >E(ne, vector<int>(5));
    int cn = 0;
    for (int j = 0; j <= (N - 1); j++) {
        for (int i = 0; i <= (N - 1); i++)
    
    {
            E[e][0] = 4;
            E[e][1] = cn;
            E[e][2] = cn + 1;
            E[e][3] = (cn + 1) + (N + 1);
            E[e][4] = cn + (N + 1);
            e = e + 1;
            cn = cn + 1;
        }
        cn = cn + 1;
    }


    // Knoten Schreiben in einem Matrix mit Name C
    for (int i = 0; i < nn; i++) {

        for (int l = 0; l < 3; l++) {

            Knoten << C[i][l] << "\t   ";
        }
        Knoten << endl;
    }
    

    // Elementen Schreiben in einem Matrix mit Name E
    for (int a = 0; a < ne; a++) {

        for (int m = 0; m < 5; m++) {

            Knoten << E[a][m] << " \t ";
        }
        Knoten << endl;
    }

    Knoten.close();
    
    return 0;
}

