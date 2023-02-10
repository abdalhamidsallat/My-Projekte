#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>







// color coding for the terminal
#define RED "\033[1;31m"
#define GREEN "\033[1;32m"
#define YELLOW "\033[1;33m"
#define BLUE "\033[1;34m"
#define MAGENTA "\033[1;35m"
#define CYAN "\033[1;36m"
#define RESET "\033[0m"







// array mit O füllen
void mkfield(int rows, int cols, char **arr) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i][j] = 'O';
        }
    }
}






//funktion zur ausgabe des arrays
void arrausgabe(int rows, int cols, char **arr) {
    printf("   ");
    for(int i = 0; i < cols; i++){
        if(i > 9)
          printf(YELLOW"%d "RESET, i);
        else
          printf(YELLOW"%d  "RESET, i);
    }
    printf("\n");
    printf("\n");
    for (int i = 0; i < rows; i++) {
        if(i > 9){
          printf(YELLOW"%d "RESET, i);
         
          }
        else{
          printf(YELLOW"%d  "RESET, i);
      
          }
        for (int j = 0; j < cols; j++) {
            printf(RED"%c  "RESET, arr[i][j]);
        }
        printf("\n");
        // comment this out if you want a more uniform looking board (it might need a bigger windows to be displayed properly)
        //printf("\n");
    }
}









// Funktion zur Speicherreservierung
void mem_alloc(int rows, int cols, char ***arr) {
    *arr = (char**)malloc(rows * sizeof(char*));
    if (*arr == NULL) {
        printf("Fehler bei der Speicherreservierung!\n");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        (*arr)[i] = (char*)malloc(cols * sizeof(char));
        if ((*arr)[i] == NULL) {
            printf("Fehler bei der Speicherreservierung!\n");
            exit(1);
        }
    }
}




// Funktion zur Auswahl der Schwierigkeit
void difficulty(int *rows, int *cols, int *mines) {
    int choice;
    while (1) {
        printf("Choose difficulty (1 - easy, 2 - medium, 3 - hard, 4 - custom): ");
        if (scanf("%d", &choice) != 1) {
            printf("Invalid input, try again.\n");
            while (getchar() != '\n'); // Clear stdin
            continue;
        }
        if (choice == 1) {
            *rows = 8;
            *cols = 8;
            *mines = 10;
            break;
        } else if (choice == 2) {
            *rows = 16;
            *cols = 16;
            *mines = 40;
            break;
        } else if (choice == 3) {
            *rows = 24;
            *cols = 24;
            *mines = 99;
            break;
        } else if (choice == 4) {
            printf("Enter number of rows and columns (rows columns): ");
            if (scanf("%d %d", rows, cols) != 2) {
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






void place_mines(int rows, int cols, int mines, char** arr){
    srand(time(NULL));
    int placed_mines = 0;
    while(placed_mines < mines){
        int x = rand() % rows;
        int y = rand() % cols;
        if (arr[x][y] != '*'){
            arr[x][y] = '*';
            placed_mines ++;
        }
    }
}






void create_hidden_array(int rows, int cols, char **arr, char ***hidden_arr) {
    mem_alloc(rows, cols, hidden_arr);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*hidden_arr)[i][j] = 'O';
        }
    }
    printf("\nHidden Array:\n");
    //arrausgabe(rows, cols, *hidden_arr);
}







// function to free memory
void free_memory(int rows, char **arr) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
        }
    free(arr);
}







// function to print ASCII art which says DEBUG that takes up the whole screen
void debug(int rows, int cols, char **arr){
    printf(GREEN "======================================================================================================\n"RESET);
    printf(GREEN "======================================================================================================\n"RESET);
    printf(GREEN "======================================================================================================\n"RESET);
    printf(RED "DDDDDDDDDDDDD      EEEEEEEEEEEEEEEEEEEEEEBBBBBBBBBBBBBBBBB   UUUUUUUU     UUUUUUUU       GGGGGGGGGGGGG\n"RESET);
    printf(RED "D::::::::::::DDD   E::::::::::::::::::::EB::::::::::::::::B  U::::::U     U::::::U    GGG::::::::::::G\n"RESET);
    printf(RED "D:::::::::::::::DD E::::::::::::::::::::EB::::::BBBBBB:::::B U::::::U     U::::::U  GG:::::::::::::::G\n"RESET);
    printf(RED "DDD:::::DDDDD:::::DEE::::::EEEEEEEEE::::EBB:::::B     B:::::BUU:::::U     U:::::UU G:::::GGGGGGGG::::G\n"RESET);                                                                                                                                        
    printf(RED "  D:::::D    D:::::D E:::::E       EEEEEE  B::::B     B:::::B U:::::U     U:::::U G:::::G       GGGGGG\n"RESET);
    printf(RED "  D:::::D     D:::::DE:::::E               B::::B     B:::::B U:::::D     D:::::UG:::::G              \n"RESET);
    printf(RED "  D:::::D     D:::::DE:::::E               B::::BBBBBB:::::B  U:::::D     D:::::UG:::::G              \n"RESET);
    printf(RED "  D:::::D     D:::::DE::::::EEEEEEEEEE     B:::::::::::::BB   U:::::D     D:::::UG:::::G    GGGGGGGGGG\n"RESET);
    printf(RED "  D:::::D     D:::::DE:::::::::::::::E     B::::BBBBBB:::::B  U:::::D     D:::::UG:::::G    G::::::::G\n"RESET);
    printf(RED "  D:::::D     D:::::DE:::::::::::::::E     B::::B     B:::::B U:::::D     D:::::UG:::::G    GGGGG::::G\n"RESET);
    printf(RED "  D:::::D     D:::::DE::::::EEEEEEEEEE     B::::B     B:::::B U:::::D     D:::::UG:::::G        G::::G\n"RESET);
    printf(RED "  D:::::D     D:::::DE:::::E       EEEEEE  B::::B     B:::::B U:::::D     D:::::UG:::::G       G:::::G\n"RESET);
    printf(RED "DDD:::::DDDDD:::::DEE::::::EEEEEEEE:::::EBB:::::BBBBBB::::::B U::::::UUUU::::::UUG:::::GGGGGGGG::::::G\n"RESET);
    printf(RED "D:::::::::::::::DD E::::::::::::::::::::EB:::::::::::::::::B   UU:::::::::::::UU  GG:::::::::::::::::G\n"RESET);
    printf(RED "D::::::::::::DDD   E::::::::::::::::::::EB::::::::::::::::B      UU:::::::::UU      GGG::::::GGG:::::G\n"RESET);
    printf(RED "DDDDDDDDDDDDD      EEEEEEEEEEEEEEEEEEEEEEBBBBBBBBBBBBBBBBB         UUUUUUUUU           GGGGGG     GGGG\n"RESET);
    printf(GREEN "======================================================================================================\n"RESET);
    printf(GREEN "======================================================================================================\n"RESET);
    printf(GREEN "======================================================================================================\n"RESET);
    arrausgabe(rows, cols, arr);
    printf(GREEN "======================================================================================================\n"RESET);
    printf(GREEN "======================================================================================================\n"RESET);
    printf(GREEN "======================================================================================================\n"RESET);
    }







int reveal(int row, int col, int rows, int cols, char **hidden_arr, char **arr) {
    if (arr[row][col] == '*') {
        hidden_arr[row][col] = '*';
        return 1;
    } else {
        int count = 0;
        for (int i = row - 1; i <= row + 1; i++) {
            for (int j = col - 1; j <= col + 1; j++) {
                if (i >= 0 && i < rows && j >= 0 && j < cols) {
                    if (arr[i][j] == '*') {
                        count++;
                    }
                }
            }
        }
        if (count == 0) {
            hidden_arr[row][col] = 'X';
        } else {
            hidden_arr[row][col] = count + '0';
        }
    }
    return 0;
}

void reveal_neighbors(int row, int col, int rows, int cols, char **hidden_arr, char **arr) {
    if (hidden_arr[row][col] == 'X') {
        for (int i = row - 1; i <= row + 1; i++) {
            for (int j = col - 1; j <= col + 1; j++) {
                if (i >= 0 && i < rows && j >= 0 && j < cols) {
                    if (hidden_arr[i][j] == 'O') {
                        int result = reveal(i, j, rows, cols, hidden_arr, arr);
                        if (result == 0) {
                            reveal_neighbors(i, j, rows, cols, hidden_arr, arr);
                        }
                    }
                }
            }
        }
    }
}

bool game_end_check(int rows, int cols, char **hidden_arr, char **arr) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(hidden_arr[i][j] == 'F') {
                if (arr[i][j] != '*') {
                    return false;
                }
            }
            if(hidden_arr[i][j] == 'O') {
                return false;
            }
        }
    }
    return true;
}


void reveal_position(int rows, int cols, char **hidden_arr, char **arr) {
    int row, col;
    while (1) {
        arrausgabe(rows,cols, hidden_arr);
        printf("Do you want to add a flag? (y/n): ");
        char flag_choice;
        scanf(" %c", &flag_choice);
        
         while(1) {
            if (!(flag_choice == 'y' || flag_choice == 'n')) {
              printf("Invalid input. Please enter 'y' or 'n': ");
              scanf(" %c", &flag_choice);
            } else {
              break;
            }
         }

        
        if(flag_choice == 'y') {
            printf("Which position should the flag be added? (row column): ");
            if (scanf("%d %d", &row, &col) == 1) {
            printf("Invalid input, try again.\n");
            while (getchar() != '\n'); // Clear stdin
            continue;
        }
            while(row < 0 || row >= rows || col < 0 || col >= cols) {
                printf("Invalid input. Please enter a valid row and column: ");
                scanf("%d %d", &row, &col);
            }
            if (hidden_arr[row][col] != 'O') {
                printf("This position has already been revealed or already has a flag.\n");
            } else {
                hidden_arr[row][col] = 'F';
            }
        } else {
            printf("Which position should be revealed? (row column): ");
            if (scanf("%d %d", &row, &col) == 1) {
            printf("Invalid input, try again.\n");
            while (getchar() != '\n'); // Clear stdin
            continue;
        }
            while(row < 0 || row >= rows || col < 0 || col >= cols) {
                printf("Invalid input. Please enter a valid row and column: ");
                scanf("%d %d", &row, &col);
            }
            if (hidden_arr[row][col] != 'O') {
                printf("This position has already been revealed or already has a flag.\n");
            } else {
                int result = reveal(row, col, rows, cols, hidden_arr, arr);
                if (result == 1) {
                    arrausgabe(rows,cols, hidden_arr);
                    printf(RED "GAME OVER, Losser. If you Lose such a simple game then I am worried about your Future\n" RESET);
                    exit(0);
                } else if (hidden_arr[row][col] == 'X') {
                    reveal_neighbors(row, col, rows, cols, hidden_arr, arr);
                }
            }
        }

        if(game_end_check(rows, cols, hidden_arr, arr)) {
            printf(GREEN"======================================================================================================\n"RESET);
            printf(GREEN"======================================================================================================\n"RESET);
            printf(GREEN"DANG IT BOY, U WON ! Now Give daddy a kiss !\n"RESET);
            printf(GREEN"======================================================================================================\n"RESET);
            printf(GREEN"======================================================================================================\n"RESET);

            exit(0);
        }
    }
}






void how_to_play() {
    printf(YELLOW "Welcome to the game of MineSweeper!\n" RESET);
    printf(CYAN "The objective of the game is to reveal all the cells that do not have a mine.\n" RESET);
    printf(GREEN "To reveal a cell, enter the row and column of the cell you want to reveal.\n" RESET);
    printf(MAGENTA "You can also add a flag to a cell by entering 'y' when prompted and then entering the cell's row and column.\n" RESET);
    printf(RED "Be careful, if you reveal a cell with a mine, the game is over.\n" RESET);
    printf(YELLOW "Good luck!\n" RESET);
}






int main() {
  how_to_play();
    int rows, cols, mines;
    difficulty(&rows, &cols, &mines);
    char **arr;
    char** hidden_arr;
    mem_alloc(rows, cols, &arr);
    mkfield(rows, cols, arr);
    place_mines(rows,cols,mines,arr);
   // debug(rows, cols, arr);
    create_hidden_array(rows, cols, arr, &hidden_arr);
    reveal_position (rows, cols, hidden_arr, arr);
   
    

    // free memory
    free_memory(rows, arr);
    free_memory(rows, hidden_arr);

    return 0;
}
