#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "interpolation.h"
#include "bmp.h"
#include "colors.h"

int main(int argn, char * argv[]){

    //Überpfüfung ob mindestens ein Startargument vorliegt:
    if (argn < 2){
        printf("Too few arguments!\n");
        exit(1);
    } else if (argn == 3){
        printf("Incomplete Input!\n");
        exit(2);
    } else if (argn > 4){
        printf("Too many arguments!\n");
        exit(3);
    }

    //Variablen für Längenangaben
    float width_data[2];
    float height_data[2];
    float x_step;
    float y_step;
    unsigned WIDTH_DATA;
    unsigned HEIGHT_DATA;
    unsigned num_element;

    //CSV-Datei öffnen und einlesen:
    FILE * datei;
    datei = fopen(argv[1], "r");

    if(datei == NULL){
        printf("Datei konnte nicht geöffnet werden.");
        exit(4);
    }

    //Einlesen der ersten beiden Zeilen, um Informationen über den Datensatz zu erhalten
    fscanf(datei, "%f%*c %f%*c %f%*c", &width_data[0], &width_data[1], &x_step);
    fscanf(datei, "%f%*c %f%*c %f%*c", &height_data[0], &height_data[1], &y_step);

    //Berechnung der enthaltenen Daten
    WIDTH_DATA = (width_data[1] - width_data[0]) / x_step + 1;
    HEIGHT_DATA = (height_data[1] - height_data[0]) / y_step + 1;
    num_element = WIDTH_DATA * HEIGHT_DATA;
    double z_min;
    double z_max;


    //Speicherplatz Allocieren
    double * data_points = (double *)malloc(num_element * sizeof(double));
    if(data_points == NULL){
        printf("Daten können nicht eingelesen werden, da nicht genügend Speicher vorhanden ist.");
        exit(5);
    }

    //Daten aus Dokument in data_points überführen
    unsigned i = 0;
    while(i < num_element && fscanf(datei, "%*f%*c %*f%*c %lf", &data_points[i]) == 1){
        i++;

    }

    fclose(datei);

    //Minimalen und Maximalen Z_Wert bestimmen
    z_min = data_points[0];
    z_max = data_points[0];
    for(unsigned i = 0; i < num_element; i++){
        if(data_points[i] < z_min){
            z_min = data_points[i];
        } else if(data_points[i] > z_max){
            z_max = data_points[i];
        }
    }

    //Funktionsaufruf, um Farbabstufungen zu ermitteln


    //Größe der Map
    unsigned WIDTH_MAP;
    unsigned HEIGHT_MAP;
    if(argn == 2){
        WIDTH_MAP = WIDTH_DATA;
        HEIGHT_MAP = HEIGHT_DATA;
    }else if(argn == 4){
        WIDTH_MAP = atoi(argv[2]);
        HEIGHT_MAP = atoi(argv[3]);
    }else{
        printf("Größe der Bitmap konnte nicht bestimmt werden!\n");
        exit(6);
    }
    unsigned num_pixel = WIDTH_MAP * HEIGHT_MAP;


    //Allocate Bitmap speicher
    double * z_map = (double *)malloc(num_pixel * sizeof(double));
    if(z_map == NULL){
        printf("Pixelmap kann nicht erstellt werden. Zu wenig Speicherplatz.");
        exit(7);
    }


    //Funktionsaufruf Interpolation
    interpolieren(data_points, z_map, WIDTH_DATA, HEIGHT_DATA, WIDTH_MAP, HEIGHT_MAP);
    free(data_points);


    //Funktionsaufruf Color_Value
    uint32_t * img = (uint32_t *)malloc(num_pixel * sizeof(uint32_t));
    if (img == NULL){
        printf("Manno, nicht genug speicher für das schöne Bild");
    }

    color_value(z_map, img, WIDTH_MAP, HEIGHT_MAP, z_min, z_max);

    free(z_map);
    bmp_create("bitmap.bmp", img, WIDTH_MAP, HEIGHT_MAP);
    free(img);
    printf("test]]]]]]]]]]]]]]]]]]]]\n" );
    return 0;
}
