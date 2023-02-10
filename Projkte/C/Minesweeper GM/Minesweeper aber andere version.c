#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define UNKNOWN_FIELD 'x'
#define MINE_FIELD '@'


// Farben fürs feld
#define RED "\033[1;31m"
#define GREEN "\033[1;32m"
#define YELLOW "\033[1;33m"
#define BLUE "\033[1;34m"
#define MAGENTA "\033[1;35m"
#define CYAN "\033[1;36m"
#define RESET "\033[0m"

void how_to_play() {
    printf(YELLOW "Welcome to the game of MineSweeper!\n" RESET);
    printf(CYAN "The objective of the game is to reveal all the cells that do not have a mine.\n" RESET);
    printf(GREEN "To reveal a cell, enter the row and column of the cell you want to reveal.\n" RESET);
    printf(MAGENTA "You can also add a flag to a cell by entering 'y' when prompted and then entering the cell's row and column.\n" RESET);
    printf(RED "Be careful, if you reveal a cell with a mine, the game is over.\n" RESET);
    printf(YELLOW "Good luck!\n" RESET);
}

void collor_number(int number) {
    switch (number) {
        case 1:
            printf(BLUE "%d" RESET, number);
            break;
        case 2:
            printf(GREEN "%d" RESET, number);
            break;
        case 3:
            printf(RED "%d" RESET, number);
            break;
        case 4:
            printf(MAGENTA "%d" RESET, number);
            break;
        case 5:
            printf(YELLOW "%d" RESET, number);
            break;
        case 6:
            printf(CYAN "%d" RESET, number);
            break;
        default:
            printf("%d", number);
            break;
    }
}

void difficulty(int *rows, int *colums, int *mines) {
    int choice;
    while (1) {
        printf(GREEN"Choose difficulty (1 - easy, 2 - medium, 3 - hard, 4 - custom):" RESET);
        if (scanf("%d", &choice) != 1) {
            printf("Invalid input, try again.\n");
            while (getchar() != '\n'); // Clear stdin
            continue;
        }
        if (choice == 1) {
            *rows = 8;
            *colums = 8;
            *mines = 10;
            break;
        } else if (choice == 2) {
            *rows = 16;
            *colums = 16;
            *mines = 40;
            break;
        } else if (choice == 3) {
            *rows = 24;
            *colums = 24;
            *mines = 99;
            break;
        } else if (choice == 4) {
            printf("Enter number of rows and columns (rows columns): ");
            if (scanf("%d %d", rows, colums) <= 2) {
                printf("Invalid input, try again.\n");
                while (getchar() != '\n'); // Clear stdin
                continue;
            }
            printf("Enter number of mines: ");
            if (scanf("%d", mines) != 1) {
                printf("Invalid input, try again.\n");
                while (getchar() != '\n'); // Clear stdin
                continue;
            }
            break;
        } else {
            printf("Invalid input, try again.\n");
        }
    }
}



//Speicher freigeben
void free_minefield( char **minefield, int rows ) {
	int i;
	//Jeden Vektor freigeben
	for (i = 0; i < rows; i++) {
    		free(minefield[i]);
   	}
   	free(minefield);
   	minefield = NULL;
}


char ** create_minefield( int rows, int columns, char init ) {
	int i, j;
	
	//Minenfeld anlegen
	char ** minefield = (char **) calloc(rows, sizeof(char*));
	
	//Falls wir keinen Speicher bekommen beenden wir das Programm
	if ( !minefield ) {
		printf( "Can't allocate memory for minefield.\n");
		exit(1);
	}
	
	
	for (i = 0; i < rows; i++) {
		minefield[i] = (char *) calloc( columns, sizeof(char) );
	}

	for(i=0; i < rows; i++) {
		for(j=0; j < columns; j++) {
			minefield[i][j] = init;
		}
	}	
	
	//Einen Zeiger auf das Minenfeld zurueckgeben
	return minefield;
}

//Zufaellig die Minen verteilen
void init_mines( char ** mines, int rows, int columns, int mines_count ) {
	int i, rand_row, rand_columns;
	
	srand(time(NULL));

	i = mines_count;

	while( i > 0 ) {
		//Zufaellige Position innerhalb der Matrix erzeugen
		rand_row = (rand()%rows);
		rand_columns = (rand()%columns);
		//Ist dort noch keine Mine?
		if( mines[rand_row][rand_columns] == 0 ) {
			//For cheaters ;-)
			//printf( "mine at %d, %d\n", rand_row, rand_columns );
			mines[rand_row][rand_columns] = 1;
			i--;
		}
	}
}

//Sind alle Minen gefunden
int check_finish( char ** minefield, char ** mines, int rows, int columns ) {
	int i, j;

	for(i=0; i < rows; i++) {
		for(j=0; j < columns; j++) {
			//Ist ein Feld noch nicht aufgedeckt welches KEINE Mine enthaelt
			//ist das Spiel noch nicht vorbei
			if( (mines[i][j] == 0) && (minefield[i][j] == UNKNOWN_FIELD) ) {
				return 0;
			}
		}
	}
	
	return 1;

}


void show_minefield( char ** minefield, int rows, int columns ) {
	system("CLS");
	int i, j;
	
	printf("\n\n");
	//Spaltenbeschriftung
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"%3d"RESET, i+1);
	}
	printf("\n");
	
	//Trennzeile
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"---"RESET);
	}
	printf("\n");
	
	
	for(i=0; i < rows; i++) {
		//Zeilenbeschriftung
		printf(BLUE"%3d|"RESET, i+1);
		for(j=0; j < columns; j++) {
			//Ausgabe des Spielfelds
			printf(YELLOW"%3c"RESET, minefield[i][j]);
		}
		//Zeilenbeschriftung
		printf(BLUE" |%-3d\n"RESET, i+1);
	}
	//Trennzeile
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"---"RESET);
	}
	printf("\n");
	//Spaltenbeschriftung	
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"%3d"RESET, i+1);
	}
	printf("\n\n");
	
}

//Minen anzeigen
void show_mines( char ** mines, int rows, int columns ) {
	int i, j;
	
	printf("\n\n");
	//Spaltenbeschriftung
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"%3d"RESET, i+1);
	}
	printf("\n");
	
	//Trennzeile
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"---"RESET);
	}
	printf("\n");
	
	for(i=0; i < rows; i++) {
		//Zeilenbeschriftung
		printf(BLUE"%3d|"RESET, i+1);
		for(j=0; j < columns; j++) {
			//Minen ausgeben
			if( mines[i][j] == 1) {
				printf(RED"  %c"RESET, MINE_FIELD);
			}
			else {
				printf(YELLOW"  %c"RESET, UNKNOWN_FIELD);
			}
		}
		//Zeilenbeschriftung
		printf(BLUE" |%-3d\n"RESET, i+1);
	}
	
	//Trennzeile
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"---"RESET);
	}
	printf("\n");
	
	//Spaltenbeschriftung
	printf("    ");
	for(i=0; i < columns; i++) {
		printf(BLUE"%3d"RESET, i+1);
	}
	printf("\n\n");	
}


int check_field( char ** minefield, char ** mines, int rows, int columns, int row, int column ) {
	char counter = 0;

	if( (row < 0) || (row >= rows) || (column < 0) || (column >= columns)) {
		printf("Keine koordiante auf dem feld!\n");
		return -1;
	}
	//An dieser Postion ist eine Mine 
	if( mines[row][column] == 1 ) {
		return 0;
	}
	
	//An der Position ist keine Mine aber wie viele gibts um mich herum?
	if( (row+1 < rows) && (mines[row+1][column] == 1) ) {
		counter++;
	}	
	if( (row-1 >= 0) && (mines[row-1][column] == 1) ) { 
		counter++;
	}
	if( (column+1 < columns) && (mines[row][column+1] == 1) ) { 
		counter++;
	}
	if( (column-1 >= 0) && (mines[row][column-1] == 1) ) {
		counter++;
	}
	if( (row+1 < rows) && (column-1 >= 0) && (mines[row+1][column-1] == 1) ) { 
		counter++;	
	}
	if( (row-1 >= 0) && (column+1 < columns) && (mines[row-1][column+1] == 1) ) { 
		counter++;
	}
	if( (row+1 < rows) && (column+1 < columns) && (mines[row+1][column+1] == 1) ) { 
		counter++;	
	}		
	if( (row-1 >= 0) && (column-1 >= 0) && (mines[row-1][column-1] == 1) ) { 
		counter++;	
	}	
	

	minefield[row][column] = 48+counter;
	
	//Ist eine Mine in der Umgebung braucht nicht weiter aufgedeckt werden
	if( counter > 0 ) {		
		return 1;
	}

	
	if( (row+1 < rows) && (minefield[row+1][column] == UNKNOWN_FIELD) && (mines[row+1][column] == 0) ) {
		check_field( minefield, mines, rows, columns, row+1, column );
	}

	if( (row-1 >= 0) && (minefield[row-1][column] == UNKNOWN_FIELD) && (mines[row-1][column] == 0) ) { 
		check_field( minefield, mines, rows, columns, row-1, column );
	}
	
	if( (column+1 < columns) && (minefield[row][column+1] == UNKNOWN_FIELD) && (mines[row][column+1] == 0) ) { 
		check_field( minefield, mines, rows, columns, row, column+1 );
	}
	
	if( (column-1 >= 0) && (minefield[row][column-1] == UNKNOWN_FIELD) && (mines[row][column-1] == 0) ) { 
		check_field( minefield, mines, rows, columns, row, column-1 );
	}

	if( (row+1 < rows) && (column-1 >= 0) && (minefield[row+1][column-1] == UNKNOWN_FIELD) && (mines[row+1][column-1] == 0) ) { 
		check_field( minefield, mines, rows, columns, row+1, column-1 );
	}

	if( (row-1 >= 0) && (column+1 < columns) && (minefield[row-1][column+1] == UNKNOWN_FIELD) && (mines[row-1][column+1] == 0) ) { 
		check_field( minefield, mines, rows, columns, row-1, column+1 );
	}

	if( (row+1 < rows) && (column+1 < columns) && (minefield[row+1][column+1] == UNKNOWN_FIELD) && (mines[row+1][column+1] == 0) ) { 
		check_field( minefield, mines, rows, columns, row+1, column+1 );
	}	
	
	if( (row-1 >= 0) && (column-1 >= 0) && (minefield[row-1][column-1] == UNKNOWN_FIELD) && (mines[row-1][column-1] == 0) ) { 
		check_field( minefield, mines, rows, columns, row-1, column-1 );
	}
		
	return 1;
}

int main(void) {
	int rows, columns, mines_count;
	int row, column;
	
    how_to_play();
	//Eingabe einlesen
	difficulty(&rows, &columns, &mines_count);

	printf("Die Spielfeldgroesse ist %d x %d, mit %d Minen\n", rows, columns, mines_count);

	//Eine array  fuer das Spielfeld
	char **minefield = create_minefield( rows, columns, UNKNOWN_FIELD );
	//Ein array fuer die Minen
	char **mines = create_minefield( rows, columns, 0 );
	
	//Die Minen in auf das Feld verteilen
	init_mines( mines, rows, columns, mines_count );
	
	//Das Spielfeld anzeigen wenn man auf eine miene tretet
	show_minefield( minefield, rows, columns );
	
	
	while(1) {

	  	printf("'Zeile Spalte' zum aufdecken eingeben (zum Beenden '0 0'):\n");
	  	if( scanf("%d %d",&row, &column) != 2 ) {	
	  		printf("Falsche Eingabe!\n");
			return 1;
		}	
		
		row--;
		column--;
		
		//Beenden 
		if( (row == -1) && (column == -1) ) {
			break;		
		}
		
		//Feldgrenzen abfragen
		if( row >= rows || row < 0 ) {
			printf("Zeile muss zwischen 1 und %d sein!\n", rows);
			continue;
		}
		if( column >= columns || column < 0 ) {
			printf("Spalte muss zwischen 1 und %d sein!\n", columns);
			continue;
		}
		
		//WEnn auf miene dann das minene feld zeigen
		if( check_field( minefield, mines, rows, columns, row, column ) == 0 ) {
			//Zeige die Minen an
			show_mines( mines, rows, columns );
			printf(RED"Zeile: %d Spalte: %d war leider eine Mine!\n"RESET, row+1, column+1 );
			break;
		}
		//wenn fertig?
		else if( check_finish(minefield, mines, rows, columns) ) {
			//Zeige die Minen an
			show_minefield( minefield, rows, columns );
			printf(GREEN"Einfach Genie!!!\n"RESET);
			break;
		}
		//Aktuelles Feld zum weiterspielen anzeigen
		else {	
			show_minefield( minefield, rows, columns );
		}
	}
	
	//Speicher freigeben	
	free_minefield( minefield, rows );
	free_minefield( mines, rows );
	
	return 1;
}




