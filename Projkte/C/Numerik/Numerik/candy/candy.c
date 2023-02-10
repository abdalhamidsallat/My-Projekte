#include <stdio.h>

#define MAX_DIM 100
unsigned W[MAX_DIM][MAX_DIM];
unsigned M[MAX_DIM][MAX_DIM];
unsigned Y, X;

void solve(int y, int x, unsigned m)
{
    if (y < 0 || x < 0 || y >= Y || x >= X) {
        return;
    }

    if (M[y][x] == 0 || m + W[y][x] < M[y][x]) {
        M[y][x] = m + W[y][x];
        //printf("%u %u -> %u\n", y, x, M[y][x]);
        solve(y, x + 1,  M[y][x]);
        solve(y, x - 1,  M[y][x]);
        solve(y + 1, x,  M[y][x]);
        solve(y - 1, x,  M[y][x]);
    }
}

int main(void)
{
    scanf("%u %u", &Y, &X);
    unsigned ye, xe, ya, xa;
    scanf("%u %u %u %u", &ye, &xe, &ya, &xa);

    for (unsigned y = 0; y < Y; y++) {
        for (unsigned x = 0; x < X; x++) {
            scanf("%u", &W[y][x]);
        }
    }

    solve(ye, xe, 0);
    printf("%u", M[ya][xa]);

    return 0;
}
