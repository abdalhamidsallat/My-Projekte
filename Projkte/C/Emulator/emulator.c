#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


// 3) Opcode indentifizierung

typedef enum opcodes
{
    MOV_addr_TO_Rn = 0b0000,
    MOV_Rn_TO_addr = 0b0001,
    MOV_MRm_TO_Rn  = 0b0010,
    MOV_Rm_TO_MRn  = 0b0011,
    MOV_Rn_literal = 0b0100,
    MOV_Rm_TO_Rn   = 0b0101,
    ADD_Rm_Rn  = 0b0110,
    SUB_Rm_Rn  = 0b0111,
    MUL_Rm_Rn  = 0b1000,
    DIV_Rm_Rn  = 0b1001,
    AND_Rm_Rn  = 0b1010,
    OR_Rm_Rn   = 0b1011,
    JZ_Rn_Rel  = 0b1100,
    CMP_Rm_Rn  = 0b1101,
    LESS_Rm_Rn = 0b1110,
    NOP        = 0b1111
} opcodes;


// 2) Umrechnung Hex -> Binär 


uint8_t hex_to_dec(char * HexArr){
    uint8_t dec = 0;
    uint8_t helper = 16;
    for(uint8_t i = 0; i < 2; i++){
        if((uint8_t)HexArr[i] >= 97 && (uint8_t)HexArr[i] <= 102){
            dec += ((uint8_t)HexArr[i]-97+10)*helper;
        }
        if(((uint8_t)HexArr[i]-48) <10){
            dec += helper*((uint8_t)HexArr[i]-48);
        }
        helper = helper / 16;
    }
    return dec;
}

void dec_to_hex(char * HexArr,uint8_t n){
    uint8_t helper[] = {15,15};
    uint8_t helper2 = 16;
    uint8_t n2 = n;
    for(uint8_t i = 0; i < 2; i++){
        while(helper[i] != 0 && n2/(helper2*helper[i]) < 1){
            helper[i]= helper[i] -1;
        }
        if(helper[i] >= 10){
            HexArr[i]= 'a'+(helper[i]-10);
        }
        if(helper[i] < 10){
            HexArr[i]= helper[i] + '0';
        }
        n2 -= 16*helper[i]; 
        helper2 = helper2 /16;
    }
}


int main(int argn, char *argv[])
{
    uint8_t Instr[16][2], Intepret[2][8];
    char Mem[256][3], Reg[16][3];
    uint8_t PC = 0;
    uint8_t PC_Max = 0;
    for(uint8_t i = 0; i < 255; i++){
        Mem[i][0]='0';
        Mem[i][1]='0';
    }
        Mem[255][0]='0';
        Mem[255][1]='0';
    for(uint8_t i = 0; i < 16; i++){
        Reg[i][0]='0';
        Reg[i][1]='0';
    }
    FILE * fl = fopen(argv[1], "r");
    if(argv[1][strlen(argv[1])-1] == 'x' && argv[1][strlen(argv[1])-2] == 'e' && argv[1][strlen(argv[1])-3] == 'h'){
        while(PC_Max < 16 && fscanf(fl, "%2x" , &Instr[PC_Max][0]) !=EOF && fscanf(fl, "%2x" , &Instr[PC_Max][1]) != EOF){
            PC_Max += 1;
        }
    } else {
        printf("Formate ausser .hex sind nicht unterstuetzt");
        return 0;
    }
    while (PC < PC_Max)
    {
        uint8_t Byte1 = Instr[PC][0];
        uint8_t OpCode = Byte1 / 0b10000;
        uint8_t Rn = Byte1 % 0b10000;
        uint8_t Byte2 = Instr[PC][1];
        uint8_t Rm = Byte2 / 0b10000;
        uint8_t adrr;
        uint8_t helper;
        
        switch (OpCode)
        {
        case MOV_addr_TO_Rn:
            for (uint8_t i = 0; i < 2; i++){
                Reg[Rn][i] = Mem[Byte2][i];
            }
            break;

        case MOV_Rn_TO_addr:
            for (uint8_t i = 0; i < 2; i++){
                Mem[Byte2][i] = Reg[Rn][i];
            }
            break;

        case MOV_MRm_TO_Rn:
            adrr = hex_to_dec(Reg[Rm]);
            for (uint8_t i = 0; i < 2; i++){
                Reg[Rn][i] = Mem[adrr][i];
            }
            break;

        case MOV_Rm_TO_MRn:
            adrr = hex_to_dec(Reg[Rn]);
            for (uint8_t i = 0; i < 2; i++){
                Mem[adrr][i] = Reg[Rm][i];
            }
            break;

        case MOV_Rn_literal:
            dec_to_hex(Reg[Rn],Byte2);
            break;

        case MOV_Rm_TO_Rn:
            for (uint8_t i = 0; i < 2; i++){
                Reg[Rn][i] = Reg[Rm][i];
            }
            break;

        case ADD_Rm_Rn:
            helper = hex_to_dec(Reg[Rn]) + hex_to_dec(Reg[Rm]);
            dec_to_hex(Reg[Rn],helper);
            break;

        case SUB_Rm_Rn:
            helper = hex_to_dec(Reg[Rn]) - hex_to_dec(Reg[Rm]);
            dec_to_hex(Reg[Rn],helper);
            break;

        case MUL_Rm_Rn:
            helper = hex_to_dec(Reg[Rn]) * hex_to_dec(Reg[Rm]);
            dec_to_hex(Reg[Rn],helper);
            break;
  
        case DIV_Rm_Rn:
            helper = hex_to_dec(Reg[Rn]) / hex_to_dec(Reg[Rm]);
            dec_to_hex(Reg[Rn],helper);
            break;

        case AND_Rm_Rn:
            if (hex_to_dec(Reg[Rn]) + hex_to_dec(Reg[Rm]) == 2){
                dec_to_hex(Reg[Rn],1);
            }else{
                dec_to_hex(Reg[Rn],0);
            }
            break;

        case OR_Rm_Rn:
            if(hex_to_dec(Reg[Rn]) + hex_to_dec(Reg[Rm]) == 0){
                dec_to_hex(Reg[Rn],0);
            }else{
                if(hex_to_dec(Reg[Rn]) + hex_to_dec(Reg[Rm]) < 3){
                dec_to_hex(Reg[Rn],1);
                }
            }
            break;

        case JZ_Rn_Rel:
            if(hex_to_dec(Reg[Rn]) == 0){
                PC =  PC + Byte2;
            }
            break;

        case CMP_Rm_Rn:
            if(hex_to_dec(Reg[Rn]) != hex_to_dec(Reg[Rm])){
                dec_to_hex(Reg[Rn],0);
            }else{
                dec_to_hex(Reg[Rn],1);
            }
            break;

        case LESS_Rm_Rn:
            if(hex_to_dec(Reg[Rn]) >= hex_to_dec(Reg[Rm])){
                dec_to_hex(Reg[Rn],0);
            }else{
                dec_to_hex(Reg[Rn],1);
            }
            break;

        case  NOP:
            //nichts passiert
            break;

            default:
            break;
        }
        PC++;
        if(atoi(argv[2])==1){
            printf("After jump to Instruction = %i:\n\nMemory:\n",PC);
            for(uint8_t i = 0; i < 255; i++){
                printf("%c%c ",Mem[i][0],Mem[i][1]);
                if(i != 0 && (i-15) % 16 == 0){
                printf("\n");
                }
            }
            printf("%c%c\n",Mem[255][0],Mem[255][1]);
            printf("\n");
            printf("Registers:\n");
            for(uint8_t i = 0; i < 16; i++){
                printf("%c%c ",Reg[i][0],Reg[i][1]);
            }
            printf("\n\n\n\n");
        }
    }
    fl = fopen("emulator.mem","w");
    for(uint8_t i = 0; i < 255; i++){
        fprintf(fl,"%c%c ",Mem[i][0],Mem[i][1]);
        if(i != 0 && (i-15) % 16 == 0){
            fprintf(fl,"\n");
        }
    }
    fprintf(fl,"%c%c\n",Mem[255][0],Mem[255][1]);
    fclose(fl);
    fl = fopen("emulator.reg","w");
    for(uint8_t i = 0; i < 16; i++){
        fprintf(fl,"%c%c ",Reg[i][0],Reg[i][1]);
    }
    fclose(fl);
}