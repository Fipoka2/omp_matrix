
#ifndef OMP_MATRIX_MATRIX_H
#define OMP_MATRIX_MATRIX_H

#include <vector>
using namespace std;

class Matrix {
    public:
        struct MatrixResult {
            double time = 0L;
            Matrix* matrix;
        };
        int** data;
        const size_t size;
        const size_t colSize;

        Matrix(size_t str, size_t col);
        Matrix(size_t s);
        static MatrixResult calculateByLinesDynamic(Matrix a, Matrix b, Matrix c);
        static MatrixResult calculateByLinesGuided(Matrix a, Matrix b, Matrix c);
        static MatrixResult calculateByLinesGuided2(Matrix a, Matrix b, Matrix c, int CHUNK_VALUE);
        static MatrixResult getMaxElemMatrix(Matrix a, Matrix b, Matrix c);
        static MatrixResult calculateByBlocks(Matrix a, Matrix b, Matrix c);
        static MatrixResult calculateByColumns(Matrix a, Matrix b, Matrix c);
        static MatrixResult calculateByLines(Matrix a, Matrix b, Matrix c);
        static void print(Matrix m);
};


#endif //OMP_MATRIX_MATRIX_H
