#include <iostream>
#include <string>
#include <omp.h>
#include <cmath>
#include "Matrix.h"
#include <limits>

struct m_time {
    double min = std::numeric_limits<double>::max();
    double average = 0;
    double max = 0;
};

const size_t MATRIX_SIZE = 6000;
const unsigned short int RUNS = 2;

int main() {
    Matrix a(MATRIX_SIZE, MATRIX_SIZE);
    Matrix b(MATRIX_SIZE, MATRIX_SIZE);
    Matrix c(MATRIX_SIZE, MATRIX_SIZE);

    m_time serial;
    m_time lines;
    m_time columns;
    m_time blocks;

    for (int i = 0; i < RUNS; i++) {
        double times[4];
        double start;
        double end;
        start = omp_get_wtime();
        Matrix res = Matrix::getMaxElemMatrix(a,b,c);
        end = omp_get_wtime();
        times[0] = end - start;
        start = omp_get_wtime();
        Matrix line = Matrix::calculateByLines(a,b,c);
        end = omp_get_wtime();
        times[1] = end - start;
        start = omp_get_wtime();
        Matrix column = Matrix::calculateByColumns(a,b,c);
        end = omp_get_wtime();
        times[2] = end - start;
        start = omp_get_wtime();
        Matrix block = Matrix::calculateByBlocks(a,b,c);
        end = omp_get_wtime();
        times[3] = end - start;

        serial.max = serial.max < times[0] ? times[0] : serial.max;
        lines.max = lines.max < times[1] ? times[1] : lines.max;
        columns.max = columns.max < times[2] ? times[2] : columns.max;
        blocks.max = blocks.max < times[3] ? times[3] : blocks.max;

        serial.min = serial.min > times[0] ? times[0] : serial.min;
        lines.min = lines.min > times[1] ? times[1] : lines.min;
        columns.min = columns.min > times[2] ? times[2] : columns.min;
        blocks.min = blocks.min > times[3] ? times[3] : blocks.min;

        serial.average += times[0];
        lines.average += times[1];
        columns.average += times[2];
        blocks.average += times[3];

    }
    serial.average /= RUNS;
    lines.average /= RUNS;
    columns.average /= RUNS;
    blocks.average /= RUNS;

    cout << "\n serial average: " << serial.average << "\n lines average: "
    << lines.average << "\n columns average: " << columns.average << "\n blocks average: "<< blocks.average;
    return 0;
}