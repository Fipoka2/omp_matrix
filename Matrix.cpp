#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include "Matrix.h"

const unsigned short int THREADS = 4;
Matrix::Matrix(size_t str, size_t col) : size(str),colSize(col)
{
    int i;
    this->data = new int*[str];
    #pragma omp parallel for private(i) num_threads(THREADS)
    for (i=0; i<str; i++)
    {
        this->data[i] = new int[col];
        for(size_t j = 0; j < col; ++j)
            this->data[i][j] = rand() % 1000;
    }
}

Matrix::Matrix(size_t s) : size(s),colSize(s)
{
    int i;
    this->data = new int*[s];
#pragma omp parallel for private(i) num_threads(THREADS)
    for (i=0; i<s; i++)
    {
        this->data[i] = new int[s];
    }
}


Matrix::MatrixResult Matrix::getMaxElemMatrix(Matrix a, Matrix b, Matrix c) {
    Matrix result(a.size, a.colSize);
    int i;
    double start = omp_get_wtime();
    for(i=0; i<result.size;i++) {
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    return {end - start, &result};
}

Matrix::MatrixResult Matrix::calculateByBlocks(Matrix a, Matrix b, Matrix c) {
    Matrix result(a.size);
    int i,j;
    double start = omp_get_wtime();
    #pragma omp parallel for private(i) shared(a,b,c,result)
    for(i=0; i<result.size;i++) {
        #pragma omp parallel for private(j)
        for(j = 0; j < result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    return {end - start, &result};
}


Matrix::MatrixResult Matrix::calculateByColumns(Matrix a, Matrix b, Matrix c) {
    Matrix result(a.size);
    int j,i,v;
    int **arr = result.data;
    double start = omp_get_wtime();
    for(i=0; i<result.size;i++) {
        #pragma omp parallel for shared(a,b,c,i) private (j)
        for(j = 0; j<result.colSize; j++) {
            arr[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    return {end - start, &result};
}

Matrix::MatrixResult Matrix::calculateByLines(Matrix a, Matrix b, Matrix c) {
    Matrix result(a.size);
    int i;
    double start = omp_get_wtime();
    #pragma omp parallel for shared(a,b,c) private(i)
    for(i=0; i<result.size;i++) {
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    return {end - start, &result};
}

void Matrix::print() {
    for (int i = 0; i < this->size; i++) {
        cout<< endl;
        for (int j = 0; j < this->colSize; i++)
            std::cout << this->data[i][j] << ' ';
    }
    cout << endl<<endl;

}

Matrix::MatrixResult Matrix::calculateByLinesDynamic(Matrix a, Matrix b, Matrix c) {
    Matrix result(a.size);
    int i;
    double start = omp_get_wtime();
#pragma omp parallel for shared(a,b,c) schedule(dynamic)
    for(i=0; i<result.size;i++) {
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    return {end - start, &result};
}

Matrix::MatrixResult Matrix::calculateByLinesGuided(Matrix a, Matrix b, Matrix c) {
    Matrix result(a.size);
    int i;
    double start = omp_get_wtime();
#pragma omp parallel for shared(a,b,c) schedule(guided, 1)
    for(i=0; i<result.size;i++) {
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    return {end - start, &result};
}

Matrix::MatrixResult Matrix::calculateByLinesGuided2(Matrix a, Matrix b, Matrix c) {
    Matrix result(a.size);
    int i;
    double start = omp_get_wtime();
#pragma omp parallel for shared(a,b,c) schedule(guided, 6)
    for(i=0; i<result.size;i++) {
        for(int j = 0; j<result.colSize; j++) {
            result.data[i][j] = std::max({a.data[i][j],b.data[i][j],c.data[i][j]});
        }
    }
    double end = omp_get_wtime();
    return {end - start, &result};
}
/*
 * 1) среднее минимальное максимальное
 * 2) по столбцам по строкам по блокам
 * 3) с разными чанками
 * 4) с разными размерностями
 * 5) с разными schedule
 * все кроме 1,2 делаются с наилучшим распаралеливанием*/