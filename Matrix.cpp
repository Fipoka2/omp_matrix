#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include "Matrix.h"

const unsigned short int THREADS = 4;

Matrix::Matrix(size_t str, size_t col) : size(str),colSize(col)
{
    int i;
    this->data.resize(str);

    #pragma omp parallel for private(i) num_threads(THREADS)
    for (i=0; i<str; i++)
    {
        this->data[i].resize(col);
        for(size_t j = 0; j < col; ++j)
            this->data[i][j] = rand() % 100;
    }
}


Matrix Matrix::getMaxElemMatrix(Matrix a, Matrix b, Matrix c) {
    cout<< "common ";
    Matrix result(a.size, a.colSize);
    int i;

    double start = omp_get_wtime();
    for(i=0; i<result.size;i++) {
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    cout << (end - start)<< endl;
    return result;
}

Matrix Matrix::calculateByBlocks(Matrix a, Matrix b, Matrix c) {
    cout<< "blocks ";
    Matrix result(a.size, a.colSize);
    int i;
    omp_set_nested(true);
    double start = omp_get_wtime();
    #pragma omp parallel for
    for(i=0; i<result.size;i++) {
        #pragma omp parallel for
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    cout << (end - start) << endl;
    return result;
}


Matrix Matrix::calculateByColumns(Matrix a, Matrix b, Matrix c) {
    cout<< "columns ";
    Matrix result(a.size, a.colSize);
    int j,i,v;

    double start = omp_get_wtime();
    for(i=0; i<result.size;i++) {
        #pragma omp parallel for  private(j)
        for(j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    cout << (end - start) << endl;
    return result;
}

Matrix Matrix::calculateByLines(Matrix a, Matrix b, Matrix c) {
    cout<< "lines ";
    Matrix result(a.size, a.colSize);
    int i;

    double start = omp_get_wtime();
    #pragma omp parallel for shared(a,b,c)
    for(i=0; i<result.size;i++) {
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    cout << (end - start) << endl;
    return result;
}
void Matrix::print() {
    for (auto str: this->data) {
        cout<< endl;
        for (auto cell: str)
            std::cout << cell << ' ';
    }
    cout << endl<<endl;

}
/*
 * 1) среднее минимальное максимальное
 * 2) по столбцам по строкам по блокам
 * 3) с разными чанками
 * 4) с разными размерностями
 * 5) с разными schedule
 * все кроме 2 делаются с наилучшим распаралеливанием*/