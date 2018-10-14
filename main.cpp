#include <iostream>
#include <string>
#include <omp.h>
#include <cmath>
#include "Matrix.h"

int main() {
    size_t s = 4000;
    size_t cl = 4000;
    Matrix a(s,cl);
    Matrix b(s,cl);
    Matrix c(s,cl);
    for (int i=0; i<10; i++) {
        cout << i << " раз" << endl;
        Matrix res = Matrix::getMaxElemMatrix(a,b,c);
        Matrix line = Matrix::calculateByLines(a,b,c);
        Matrix column = Matrix::calculateByColumns(a,b,c);
        Matrix block = Matrix::calculateByBlocks(a,b,c);
    }
    return 0;
}