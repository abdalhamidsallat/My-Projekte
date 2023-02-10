#include <stdio.h>
#include <stdlib.h>
// #include <time.h>
#include <string.h>

typedef struct spieler{// structur
  char name[50];
}SP;

void spieler_install(SP *spieler){// Spieler Daten Eingabe
  printf("Spierlname: ");
  scanf("%s", spieler->name);

  printf("-----------\n");
}

void printer(SP* list) { //Spieler Daten Auf bildschirm ausgeben!

  printf(" %s spieler ",list->name);

}

SP* studentalloc(){// Dynamisch Speicher
  SP *P;
  P =(SP*)malloc(sizeof(SP));
  return P;
}

void Spielern_namen() {
//  system("CLS");
  SP* spielr1=studentalloc();// spieler Dynamisch defnieren // Sp* spieler1 =(SP*)malloc (sizeof(SP));
  SP* spieler2=studentalloc();
  spieler_install(spielr1);//spieler Dynamisch installieren// namen eingeben
  spieler_install(spieler2);
  system("CLS");
  printer(spielr1); //Spieler Daten Auf bildschirm ausgeben!
  printf("  VS   ");
  printer(spieler2);
  for (int i=0;i<9999999999;i++) { // Pause
    /* code */
  }
}
void bildstart() {
  printf(".....................................................................................\n");
  printf("......oooooo.....ooooooo.....ooooo......ooooooo.....ooooooo.....ooooooo...o.....o....\n");
  printf(".....o..............0.......o.....o.....o......o.......0........0.........o.o...o....\n");
  printf("......oooooo........0.......ooooooo.....ooooooo........0........oooooo....o..o..o....\n");
  printf("............o.......0.......o.....o.....o..o...........0........0.........0...o.o....\n");
  printf("......oooooo........0.......o.....o.....o....o.........0........ooooooo...o.....o....\n");
  printf(".....................................................................................\n");
  for (int i=0;i<399999999;i++) { // Pause
    /* code */
  }
}
void teimer() {
  printf(".....................................................................................\n");
  printf("................................ooooooo..............................................\n");
  printf("......................................0..............................................\n");
  printf("................................ooooooo..............................................\n");
  printf("......................................0..............................................\n");
  printf("................................ooooooo..............................................\n");
  printf(".....................................................................................\n");
  for (int i=0;i<399999999;i++) { // Pause
    /* code */
  }
  system("CLS");
  printf(".....................................................................................\n");
  printf("................................ooooooo..............................................\n");
  printf("......................................o..............................................\n");
  printf("................................ooooooo..............................................\n");
  printf("................................o....................................................\n");
  printf("................................ooooooo..............................................\n");
  printf(".....................................................................................\n");
  for (int i=0;i<399999999;i++) {
    /* code */
  }
  system("CLS");
  printf(".....................................................................................\n");
  printf("....................................o0...............................................\n");
  printf(".....................................0...............................................\n");
  printf(".....................................0...............................................\n");
  printf(".....................................0...............................................\n");
  printf(".....................................0...............................................\n");
  printf(".....................................................................................\n");
  for (int i=0;i<399999999;i++) {
    /* code */
  }
  system("CLS");
  printf(".....................................................................................\n");
  printf("......0........0000000....000000000.....0000000............00000000...... 00000......\n");
  printf("......0........0..............0........0..................0..............0.....0.....\n");
  printf("......0........000000.........0.........0000000...........0....0000......0.....0.....\n");
  printf("......0........0..............0................0..........0.. .....0.....0.....0.....\n");
  printf("......00000....0000000........0.........0000000............00000000.......00000......\n");
  printf(".....................................................................................\n");
  for (int i=0;i<799999999;i++) {
    /* code */
  }

}

// spiel-code
char Matrix[3][3] = {
                     {'1','2','3'},
                     {'4','5','6'},
                     {'7','8','9'}
                   };
static char Player ='X';

char pos=0;
void PrintMatrx() {

  system("CLS");
  printf("+---+---+---+\n");
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
     printf("| %c ",Matrix[i][j]);
    }
    printf("|\n");
    printf("+---+---+---+\n");

  }
}
void change(char *Player){// spieler tauschen!
  if(*Player == 'X')
     *Player='O';
  //printf("Chose a Psition - (%c)\n",Player);
  else{
     *Player = 'X';
  }
}
void Play(char player) { // eine position der schpieler auswählenf, Player ist: entweder X oder O
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      if(Matrix[i][j]==pos){
       Matrix[i][j]=Player;
     }
    }
  }

}

char WhoWin() {// contorl structur für das spiel und wer gewinnt!
  int Xc=0;int Oc=0;
  int counter=0;// zähler, wie viele frei sind noch im Box
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      if (Matrix[i][j] != 'X'&& Matrix[i][j]!='O')
      {
        counter++;
      }
    }

  }
  printf("\ncounter isn %d\n",counter );
  if (counter == 0) return 'Z';
  counter = 0;

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      if (Matrix[i][j] == 'X')Xc++;
      else if (Matrix[i][j] == 'O')Oc++;
      if (Xc==3||Oc==3){
        if( Xc > Oc ) return 'X';
        else{return 'O';}
      }
    }
    Xc=0;
    Oc=0;
  }

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      if (Matrix[j][i] == 'X')Xc++;
      else if (Matrix[j][i] == 'O')Oc++;
      if (Xc==3||Oc==3){
        if( Xc > Oc ) return 'X';
        else{return 'O';}
      }
    }
    Xc=0;
    Oc=0;
  }
  if(Matrix[0][0]=='X'&&Matrix[1][1]=='X'&&Matrix[2][2]=='X'){
    return 'X';
  }
  else if(Matrix[0][0]=='O'&&Matrix[1][1]=='O'&&Matrix[2][2]=='O'){
    return 'O';
  }
  else if(Matrix[0][2]=='X'&&Matrix[1][1]=='X'&&Matrix[2][0]=='X'){
    return 'X';
  }
  else if(Matrix[0][2]=='O'&&Matrix[1][1]=='O'&&Matrix[2][0]=='O'){
      return 'O';
    }

  return '.';
}
int gegencomputer(int pos) {
  if(!pos){
    int random =1+rand()&9;
    return random;
  }
}
int main(){
  int starten1=0;
  int starten2=0;
  printf("Mochtest du Tik Tak To speilen:) aber sehr gerne\n\n\t1)->zum Starten druck (1):\n");
  scanf("%d",&starten1);
  system("CLS");
  if(starten1==1){
    printf("Die Regeln sind so einfach: \n\t");
    printf("1) der spieler X fangt immer an \n\t");
    printf("2)spiler O kommt immer der naxt\n\t");
    printf("3)der spiler giwinnt wenn entweder \n\t\t");
    printf("->reihe\n\t\t");
    printf("->spalte \n\t\t");
    printf("->oder diagonal \n\t");
    printf("X oder O reicht\n");
    printf("Um die spiel zu beginnen drucken Sie auf (2)\n");
    printf("Um die spiel zu beenden drucken Sie auf (3)\n");
    scanf("%d",&starten2);
    system("CLS");
  //  bildstart();
    Spielern_namen();
    for (int i=0;i<599999999;i++) {
      /* code */
    }
    system("CLS");
    teimer();
  }


  switch (starten2) {
    case 2: printf("Das spiel beginnt mit player 'X'\n");
            PrintMatrx();
            while (WhoWin() == '.'){
              printf("Chose a Psition Spieler (%c)\n",Player);
              //pos=0;
              scanf(" %c",&pos);
              Play(Player);
              PrintMatrx();
              change(&Player);
            }
            if(WhoWin()=='Z')
              printf("Nobody has win\n");
            else {
              PrintMatrx();
              printf("You Winnnnnnn %c\n",WhoWin());
            }
            break;
    case 3:printf("\n:( Spielende\n\n\n");break;

    default:break;

  }
   return 0;
}
