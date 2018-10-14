
#ifndef OMP_MATRIX_MATRIX_H
#define OMP_MATRIX_MATRIX_H

#include <vector>
using namespace std;

class Matrix {
    public:
        vector< vector<int> > data;
        const size_t size;
        const size_t colSize;

        Matrix(size_t str, size_t col);
        static Matrix getMaxElemMatrix(Matrix a, Matrix b, Matrix c);
        static Matrix calculateByBlocks(Matrix a, Matrix b, Matrix c);
        static Matrix calculateByColumns(Matrix a, Matrix b, Matrix c);
        static Matrix calculateByLines(Matrix a, Matrix b, Matrix c);
        void print();
};


#endif //OMP_MATRIX_MATRIX_H
