int interpolieren(double *data_points, double *map, unsigned WIDTH_DATA, unsigned HEIGHT_DATA, unsigned WIDTH_MAP, unsigned HEIGHT_MAP){

    double proportion_x, proportion_y;
    unsigned smaller_x, smaller_y, bigger_x, bigger_y; //Punkte in data_points[]
    double pos_y_l, pos_y_r;

    for (unsigned y = 0; y < HEIGHT_MAP ; y++){
        for (unsigned x = 0; x < WIDTH_MAP ; x++){

            proportion_x = ((double) x / WIDTH_MAP) * (WIDTH_DATA - 1);
            smaller_x = proportion_x;
            
            if ( (double)(proportion_x +1) == (int) (proportion_x +1)){
                bigger_x = proportion_x;
            }else{
                bigger_x = proportion_x +1;
            }
            proportion_x -= smaller_x;


            proportion_y = ((double) y / HEIGHT_MAP) * (HEIGHT_DATA -1);
            smaller_y = proportion_y;

            if ( (double)(proportion_y +1) == (int) (proportion_y +1)){
                bigger_y = proportion_y;
            }else{
                bigger_y = proportion_y +1;
            }
            proportion_y -= smaller_y;


            pos_y_l = (data_points[smaller_x * HEIGHT_DATA + smaller_y]
                    +
                    (data_points[smaller_x * HEIGHT_DATA + bigger_y] - data_points[smaller_x * HEIGHT_DATA + smaller_y]) * proportion_y);
            pos_y_r = (data_points[bigger_x * HEIGHT_DATA + smaller_y]
                    +
                    (data_points[bigger_x * HEIGHT_DATA + bigger_y] - data_points[bigger_x * HEIGHT_DATA + smaller_y]) * proportion_y);

            map[y * WIDTH_MAP + x] = (pos_y_l + (pos_y_r - pos_y_l) * proportion_x);

        }
    }
    return 0;
}