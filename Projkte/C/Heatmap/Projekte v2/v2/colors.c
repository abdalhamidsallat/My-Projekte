#include <stdint.h>
#include <math.h>
#include <stdio.h>


int color_cluster(double min, double max, double * start_values, int steps){ //berechnet Werte, ab denen Farbstufe beginnt
    
    double difference = max - min;
    for(int i = 0; i < steps; i++){
        start_values[i] = min + (i  * difference / steps);
    }

    return 0;
}

void color_value(double * map, uint32_t * img, int width, int height, double min, double max){ //weist z-Werten aus map[] Farbwerte in img[] zu

    
    //Definition der Anzahl der Farbabstufungen
    #define COLOR_DEPTH 10
    
    //Array, in dem die Start-Z-Werte für eine Farbstufe gespeichert werden
    double start_values[COLOR_DEPTH];

    //Farben zum berechnen des Farbverlaufs
    int color_deep[] = {240, 0, 0};     //Blau
    int color_medium[] = {0, 240, 0};   //Grün
    int color_high[] = {0, 0, 240};     //Rot
    int color_tmp[3];
    
    //Aufrufen der Color_Cluster-Funktion
    color_cluster(min, max, start_values, COLOR_DEPTH);

    //temporäre Variable zur Einordnung der Z-Werte in Farbstufen
    int depth;

    //Für jeden Wert im Map-Array
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            
            //wird der Wert eriner Farbstufe zugeordnet
            depth = 1;
            while(map[y * width + x] > start_values[depth] && depth < COLOR_DEPTH){ //
                depth++;
            }

            //Wert des Pixels wird zur Reinigung auf Null gesetzt
            img[y * width + x] = 0;

            //Falls der Wert der Map zur oberen Hälfte der Z_Werte gehört, wird ein Ferbwert zwischen Grün und Rot ermittelt
            if(depth >= (COLOR_DEPTH / 2)){

                for(unsigned i = 0; i < 3; i++){
                    color_tmp[i] = color_medium[i] + ((2 * ((double)depth / COLOR_DEPTH)) - 1) * (color_high[i] - color_medium[i]);
                    img[y * width + x] += color_tmp[i] * pow(2, i * 8);
                }

            //Falls der Wert der Map zur unteren Hälfte der Z_Werte gehört, wird ein Ferbwert zwischen Blau und Grün ermittelt
            }else{

                for(unsigned i = 0; i < 3; i++){
                    color_tmp[i] = color_deep[i] + (2 * ((double)depth / COLOR_DEPTH)) * (color_medium[i] - color_deep[i]);
                    img[y * width + x] += color_tmp[i] * pow(2, i * 8);
                }
                
            }
        }
    }
}
