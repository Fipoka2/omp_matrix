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

const size_t MATRIX_SIZE = 400;
const unsigned short int RUNS = 5;

int main() {
    omp_set_nested(true);
    Matrix a(MATRIX_SIZE, MATRIX_SIZE);
    Matrix b(MATRIX_SIZE, MATRIX_SIZE);
    Matrix c(MATRIX_SIZE, MATRIX_SIZE);

    m_time serial;
    m_time lines;
    m_time columns;
    m_time blocks;

    for (int i = 0; i < RUNS; i++) {
        double times[4];
        Matrix::MatrixResult res = Matrix::getMaxElemMatrix(a,b,c);
        times[0] = res.time;
        Matrix::MatrixResult line = Matrix::calculateByLines(a,b,c);
        times[1] = line.time;
        Matrix::MatrixResult column = Matrix::calculateByColumns(a,b,c);
        times[2] = column.time;
        Matrix::MatrixResult block = Matrix::calculateByBlocks(a,b,c);
        times[3] = block.time;

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

    Matrix::MatrixResult linesDynamic = Matrix::calculateByLinesDynamic(a,b,c);
    Matrix::MatrixResult linesG = Matrix::calculateByLinesDynamic(a,b,c);
    Matrix::MatrixResult linesG2 = Matrix::calculateByLinesDynamic(a,b,c);

    cout << "\n serial average: " << serial.average << "\n lines average: "
    << lines.average << "\n columns average: " << columns.average << "\n blocks average: "<< blocks.average;

    cout << "\n\n serial min: " << serial.min << "\n lines min: "
         << lines.min << "\n columns min: " << columns.min << "\n blocks min: "<< blocks.min;

    cout << "\n\n serial max: " << serial.max << "\n lines max: "
         << lines.max << "\n columns max: " << columns.max << "\n blocks max: "<< blocks.max;

    cout<< "\nlines dynamic " << linesDynamic.time;
    cout<< "\nlines guided chunk " << linesG.time;
    cout<< "\nlines guided chunk " << linesG2.time;
    return 0;
}