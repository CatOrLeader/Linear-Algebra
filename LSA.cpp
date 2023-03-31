//
// Created by Mukhutdinov Artur Robertovich on 30.03.2023.
// a.mukhutdinov@innopolis.university
//

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#define errorMessage "Error: the dimensional problem occurred"

using namespace std;

// Exceptions

class ZeroDet : exception {
public:
    [[nodiscard]] const char * what() const noexcept override {
        return "The method is not applicable!";
    }
};

class NotConverges : exception {
public:
    [[nodiscard]] const char * what() const noexcept override {
        return "The method is not applicable!";
    }
};


// Matrix classes

/*Implements behaviour of the Matrix*/
class Matrix {
protected:
    int rowCount;
    int columnCount;
    double** matrix;
public:
    Matrix(int row0, int column0) {
        rowCount = row0;
        columnCount = column0;

        auto** temp = new double*[rowCount];
        for (int i = 0; i < rowCount; i++) {
            temp[i] = new double[columnCount];
        }

        matrix = temp;
    }

    Matrix(const Matrix& srcMatrix) {
        rowCount = srcMatrix.getRow();
        columnCount = srcMatrix.getColumn();

        auto** temp = new double*[rowCount];
        for (int i = 0; i < rowCount; i++) {
            temp[i] = new double[columnCount];
        }

        for (int i = 0; i < rowCount; i++) {
            for (int j = 0; j < columnCount; j++) {
                temp[i][j] = srcMatrix.getMatrix()[i][j];
            }
        }

        matrix = temp;
    }

    friend istream& operator>> (istream &input, Matrix const &M) {
        for (int i = 0; i < M.rowCount; i++) {
            for (int j = 0; j < M.columnCount; j++) {
                input >> M.matrix[i][j];
            }
        }
        return input;
    }

    /*Precision of float-type number - 2 digit after the point*/
    friend ostream& operator<< (ostream &out, Matrix const &M) {
        out << fixed << showpoint;
        out << setprecision(4);

        for (int i = 0; i < M.rowCount; i++) {
            for (int j = 0; j < M.columnCount; j++) {
                out << (fabs(M.matrix[i][j]) < 0.00005 ? 0.00 : M.matrix[i][j]) << " ";
            }
            out << "\n";
        }
        return out;
    }

    Matrix& operator= (Matrix const &beCopied) {
        if (this != &beCopied) {
            rowCount = beCopied.getRow();
            columnCount = beCopied.getColumn();

            auto** temp = new double*[rowCount];
            for (int i = 0; i < rowCount; i++) {
                temp[i] = new double[columnCount];
            }

            for (int i = 0; i < rowCount; i++) {
                for (int j = 0; j < columnCount; j++) {
                    temp[i][j] = beCopied.getMatrix()[i][j];
                }
            }

            matrix = temp;
        }
        return *this;
    }

    [[nodiscard]] Matrix transpose() const {
        Matrix transposed(this->getColumn(), this->getRow());

        for (int i = 0; i < transposed.getRow(); i++) {
            for (int j = 0; j < transposed.getColumn(); j++) {
                transposed.getMatrix()[i][j] = this->getMatrix()[j][i];
            }
        }

        return transposed;
    }

    // Get count of rows
    [[nodiscard]] int getRow() const {
        return rowCount;
    }

    // Get count of columns
    [[nodiscard]] int getColumn() const {
        return columnCount;
    }

    [[nodiscard]] double **getMatrix() const {
        return matrix;
    }
};

Matrix operator+ (Matrix const& M1, Matrix const& M2) {
    if (M1.getRow() != M2.getRow() || M1.getColumn() != M2.getColumn()) {
        cout << errorMessage << endl;
        return {0, 0};
    }

    Matrix result(M1.getRow(), M1.getColumn());

    for (int i = 0; i < M1.getRow(); i++) {
        for (int j = 0; j < M1.getColumn(); j++) {
            result.getMatrix()[i][j] = M1.getMatrix()[i][j] + M2.getMatrix()[i][j];
        }
    }

    return result;
}

Matrix operator- (Matrix const& M1, Matrix const& M2) {
    if (M1.getRow() != M2.getRow() || M1.getColumn() != M2.getColumn()) {
        cout << errorMessage << endl;
        return {0, 0};
    }

    Matrix result(M1.getRow(), M1.getColumn());

    for (int i = 0; i < M1.getRow(); i++) {
        for (int j = 0; j < M1.getColumn(); j++) {
            result.getMatrix()[i][j] = M1.getMatrix()[i][j] - M2.getMatrix()[i][j];
        }
    }

    return result;
}

Matrix operator* (Matrix const& M1, Matrix const& M2) {
    if (M1.getColumn() != M2.getRow()) {
        cout << errorMessage << endl;
        return {0, 0};
    }

    Matrix result(M1.getRow(), M2.getColumn());

    for (int i = 0; i < result.getRow(); i++) {
        for (int j = 0; j < result.getColumn(); j++) {
            result.getMatrix()[i][j] = 0;
            for (int k = 0; k < M1.getColumn(); k++) {
                result.getMatrix()[i][j] += M1.getMatrix()[i][k] * M2.getMatrix()[k][j];
            }
        }
    }

    return result;
}

class SquaredMatrix : public Matrix {
public:
    explicit SquaredMatrix(int size0) : Matrix(size0, size0) {};
};

class IdentityMatrix : public SquaredMatrix {
public:
    explicit IdentityMatrix(int size0) : SquaredMatrix(size0) {
        for (int i = 0; i < rowCount; i++) {
            for (int j = 0; j < columnCount; j++){
                if (i == j) {
                    matrix[i][j] = 1;
                    continue;
                }

                matrix[i][j] = 0;
            }
        }
    }
};

class EliminationMatrix : public IdentityMatrix {
public:
    explicit EliminationMatrix(int size0, int rowPos, int colPos, double value) : IdentityMatrix(size0) {
        matrix[rowPos][colPos] = value;
    }
};

class PermutationMatrix : public IdentityMatrix {
public:
    explicit PermutationMatrix(int size0) : IdentityMatrix(size0) { };

    void exchangeRows(int rowNum1, int rowNum2) {
        matrix[rowNum1][rowNum1] = 0;
        matrix[rowNum1][rowNum2] = 1;

        matrix[rowNum2][rowNum2] = 0;
        matrix[rowNum2][rowNum1] = 1;
    }
};

// Additional_processes

/*Get matrix with size [matrix.getRow() x matrix.getColumn() * 2); This matrix includes on the left side original
 matrix, and on the right side - identity matrix*/
Matrix getAugmentedMatrix(const Matrix& matrix, int* step) {
    Matrix augmentedMatrix(matrix.getRow(), matrix.getColumn() * 2);

    for (int i = 0; i < matrix.getRow(); i++) {
        for (int j = 0; j < matrix.getColumn(); j++) {
            // Copy source matrix
            augmentedMatrix.getMatrix()[i][j] = matrix.getMatrix()[i][j];

            // Make on the right identity matrix
            if (i == j) {
                augmentedMatrix.getMatrix()[i][j + matrix.getColumn()] = 1;
            } else {
                augmentedMatrix.getMatrix()[i][j + matrix.getColumn()] = 0;
            }
        }
    }

//    cout << "step #" << *step << ": Augmented Matrix\n" << augmentedMatrix;
    (*step)++;
    return augmentedMatrix;
}

// Elimination Processes

/*In matrix: row1 - times * row2*/
void rowSubtraction(Matrix& matrix, int rowNumToBeSubtracted, int rowNumToSubtract, double times) {
    for (int i = 0; i < matrix.getColumn(); i++) {
        matrix.getMatrix()[rowNumToBeSubtracted][i] -= matrix.getMatrix()[rowNumToSubtract][i] * times;
    }
}

/*Convert the matrix into upper triangular form and sequentially output Elimination matrices + matrix itself*/
void eliminationProcess(Matrix& matrix) {
    for (int j = 0; j < matrix.getColumn() - 1; j++) {
        for (int i = 1; i < matrix.getRow(); i++) {
            double times = matrix.getMatrix()[i][j] / matrix.getMatrix()[j][j];

            EliminationMatrix tempEliminationMatrix(matrix.getRow(), i, j, -times);
            rowSubtraction(matrix, i, j, times);

            cout << tempEliminationMatrix;
            cout << matrix;
        }
    }
}

// Determinant

/*Find index of row, which need to be swapped with the current upper row*/
int permutationIndex(const Matrix& matrix, int currentUpperRow, int currentCol) {
    double max = fabs(matrix.getMatrix()[currentUpperRow][currentCol]);
    int maxInd = -1;

    for (int i = currentUpperRow; i < matrix.getRow(); i++) {
        if (max < (fabs(matrix.getMatrix()[i][currentCol]))) {
            max = fabs(matrix.getMatrix()[i][currentCol]);
            maxInd = i;
        }
    }

    return maxInd;
}

double findDeterminant(Matrix matrix) {
    int step = 1;

    for (int j = 0; j < matrix.getColumn() - 1; j++) {
        bool permutationFlag = false;
        for (int i = j + 1; i < matrix.getRow(); i++) {
            if (permutationIndex(matrix, j, j) != -1 && !permutationFlag) {
                permutationFlag = true;

                PermutationMatrix P(matrix.getRow());
                P.exchangeRows(j, permutationIndex(matrix, j, j));

                matrix = P * matrix;

//                cout << "step #" << step << ": permutation" << endl;
//                cout << matrix;
                step++;
            }

            double times = matrix.getMatrix()[i][j] / matrix.getMatrix()[j][j];

            rowSubtraction(matrix, i, j, times);

//            cout << "step #" << step << ": elimination" << endl;
//            cout << matrix;
            step++;
        }
    }

    double det = 1;
    for (int i = 0; i < matrix.getRow(); i++) {
        det *= matrix.getMatrix()[i][i];
    }

    return det;
}

// Inverse Matrix

/*Eliminate left side of Augmented matrix into upper triangular form with the sequential output of matrix states*/
Matrix directWay(Matrix matrix, int sizeOfOriginalMatrix, int* step) {
//    cout << "Direct way:\n";

    for (int j = 0; j < sizeOfOriginalMatrix - 1; j++) {
        bool permutationFlag = false;
        for (int i = j + 1; i < sizeOfOriginalMatrix; i++) {
            if (permutationIndex(matrix, j, j) != -1 && !permutationFlag) {
                permutationFlag = true;

                PermutationMatrix P(matrix.getRow());
                P.exchangeRows(j, permutationIndex(matrix, j, j));

                matrix = P * matrix;

//                cout << "step #" << *step << ": permutation" << endl;
//                cout << matrix;
                (*step)++;
            }

            double times = matrix.getMatrix()[i][j] / matrix.getMatrix()[j][j];

            if (times == 0) {
                continue;
            }

            rowSubtraction(matrix, i, j, times);

//            cout << "step #" << *step << ": elimination" << endl;
//            cout << matrix;
            (*step)++;
        }
    }

    return matrix;
}

/*Eliminate left side of Augmented matrix into diagonal form with the sequential output of matrix states*/
Matrix wayBack(Matrix matrix, int sizeOfOriginalMatrix, int* step) {
//    cout << "Way back:\n";

    for (int j = sizeOfOriginalMatrix - 1; j > 0; j--) {
        for (int i = j - 1; i >= 0; i--) {
            double times = matrix.getMatrix()[i][j] / matrix.getMatrix()[j][j];

            if (times == 0) {
                continue;
            }

            rowSubtraction(matrix, i, j, times);
//            cout << "step #" << *step << ": elimination\n";
//            cout << matrix;
            (*step)++;
        }
    }

    return matrix;
}

/*Normalize every row of augmented matrix*/
Matrix normalisation(const Matrix& matrix, int sizeOfOriginalMatrix) {
//    cout << "Diagonal normalization:\n";

    for (int i = 0; i < sizeOfOriginalMatrix; i++) {
        double normalizeFactor = matrix.getMatrix()[i][i];
        for (int j = 0; j < matrix.getColumn(); j++) {
            matrix.getMatrix()[i][j] /= normalizeFactor;
        }
    }

//    cout << matrix;
    return matrix;
}

Matrix getInverseMatrix(Matrix& matrix) {
    int step = 0;
    int sizeOfOriginalMatrix = matrix.getRow();
    Matrix augmentedMatrix = getAugmentedMatrix(matrix, &step);
    Matrix upperTriangularMatrix = directWay(augmentedMatrix, sizeOfOriginalMatrix, &step);
    Matrix diagonalMatrix = wayBack(upperTriangularMatrix, sizeOfOriginalMatrix, &step);
    Matrix normalised = normalisation(diagonalMatrix, sizeOfOriginalMatrix);

    SquaredMatrix inverseMatrix(sizeOfOriginalMatrix);

    for (int i = 0; i < matrix.getRow(); i++) {
        for (int j = 0; j < sizeOfOriginalMatrix; j++) {
            inverseMatrix.getMatrix()[i][j] = normalised.getMatrix()[i][j + sizeOfOriginalMatrix];
        }
    }

    return inverseMatrix;
}

// ColumnVector

/*Implements behaviour of the Column Vector*/
class ColumnVector {
private:
    int dimension;
    double* vector;
public:
    explicit ColumnVector(int dimension0) {
        dimension = dimension0;

        vector = new double[dimension];
    };

    explicit ColumnVector() {
        dimension = 0;
        vector = nullptr;
    }

    friend istream& operator>>(istream& input, ColumnVector& vec) {
        for (int i = 0; i < vec.getDimension(); i++) {
            input >> vec.getVector()[i];
        }

        return input;
    }

    /*Precision of float-type number - 2 digit after the point*/
    friend ostream& operator<<(ostream& output, ColumnVector& vec) {
        output << fixed << showpoint;
        output << setprecision(4);

        for (int i = 0; i < vec.getDimension(); i++) {
            output << (fabs(vec.getVector()[i]) < 0.00005 ? 0.00 : vec.getVector()[i]) << "\n";
        }

        return output;
    }

    friend ColumnVector operator+(ColumnVector& vec1, ColumnVector& vec2) {
        if (vec1.getDimension() != vec2.getDimension()) {
            cout << errorMessage << endl;
            return ColumnVector();
        }

        ColumnVector result(vec1.getDimension());

        for (int i = 0; i < result.getDimension(); i++) {
            result.getVector()[i] = vec1.getVector()[i] + vec2.getVector()[i];
        }
        return result;
    }

    friend ColumnVector operator-(ColumnVector& vec1, ColumnVector& vec2) {
        if (vec1.getDimension() != vec2.getDimension()) {
            cout << errorMessage << endl;
            return ColumnVector();
        }

        ColumnVector result(vec1.getDimension());

        for (int i = 0; i < result.getDimension(); i++) {
            result.getVector()[i] = vec1.getVector()[i] - vec2.getVector()[i];
        }
        return result;
    }

    friend ColumnVector operator*(ColumnVector& vec1, ColumnVector& vec2) {
        if (vec1.getDimension() != vec2.getDimension()) {
            cout << errorMessage << endl;
            return ColumnVector();
        }

        ColumnVector result(vec1.getDimension());

        for (int i = 0; i < result.getDimension(); i++) {
            result.getVector()[i] = vec1.getVector()[i] * vec2.getVector()[i];
        }
        return result;
    }

    ColumnVector& operator= (ColumnVector const &beCopied) {
        if (this != &beCopied) {
            dimension = beCopied.getDimension();

            auto* temp = new double[dimension];

            for (int i = 0; i < dimension; i++) {
                temp[i] = beCopied.getVector()[i];

            }

            vector = temp;
        }
        return *this;
    }

    void exchangeRows(int rowNum1, int rowNum2) {
        double temp = vector[rowNum1];
        vector[rowNum1] = vector[rowNum2];
        vector[rowNum2] = temp;
    }

    /*In vector: row1 - times * row2*/
    void rowSubtraction(int rowNumToBeSubtracted, int rowNumToSubtract, double times) {
        vector[rowNumToBeSubtracted] -= vector[rowNumToSubtract] * times;
    }

    double getNorm() {
        double answer = 0;
        for (int i = 0; i < dimension; i++) {
            answer += pow(vector[i], 2);
        }

        return sqrt(answer);
    }

    /*Get count of dimensions*/
    [[nodiscard]] int getDimension() const {
        return dimension;
    }

    [[nodiscard]] double *getVector() const {
        return vector;
    }
};

// Solving Linear System

/*Eliminate matrix into upper triangular form with the free coefficients vector;
 *Output sequentially matrix's state and free coefficients vector's state*/
void directWay(Matrix& matrix, ColumnVector& vec, int* step) {
    for (int j = 0; j < matrix.getColumn() - 1; j++) {
        bool permutationFlag = false;
        for (int i = j + 1; i < matrix.getRow(); i++) {
            if (permutationIndex(matrix, j, j) != -1 && !permutationFlag) {
                permutationFlag = true;

                PermutationMatrix P(matrix.getRow());

                // Permutation for vector with free coefficients
                vec.exchangeRows(j, permutationIndex(matrix, j, j));

                // Permutation for matrix
                P.exchangeRows(j, permutationIndex(matrix, j, j));
                matrix = P * matrix;

                // Output
//                cout << "step #" << *step << ": permutation\n";
//                cout << matrix << vec;
                (*step)++;
            }

            if (matrix.getMatrix()[j][j] == 0) {
                continue;
            }

            double times = matrix.getMatrix()[i][j] / matrix.getMatrix()[j][j];

            if (times == 0) {
                continue;
            }

            rowSubtraction(matrix, i, j, times);
            vec.rowSubtraction(i, j, times);

//            cout << "step #" << *step << ": elimination" << endl;
//            cout << matrix << vec;
            (*step)++;
        }
    }
}

/*Eliminate matrix into diagonal form with the free coefficients vector;
 *Output sequentially matrix's state and free coefficients vector's state*/
void wayBack(Matrix& matrix, ColumnVector& vec, int* step) {
    for (int j = matrix.getColumn() - 1; j > 0; j--) {
        for (int i = j - 1; i >= 0; i--) {
            if (matrix.getMatrix()[j][j] == 0) {
                continue;
            }

            double times = matrix.getMatrix()[i][j] / matrix.getMatrix()[j][j];

            if (times == 0) {
                continue;
            }

            rowSubtraction(matrix, i, j, times);
            vec.rowSubtraction(i, j, times);

//            cout << "step #" << *step << ": elimination" << endl;
//            cout << matrix << vec;
            (*step)++;
        }
    }
}

/*Normalise every row of matrix;
 *Output sequentially matrix's state and free coefficients vector's state*/
void normalisation(Matrix& matrix, ColumnVector& vec) {
//    cout << "Diagonal normalization:\n";

    for (int i = 0; i < matrix.getRow(); i++) {
        double normalizationFactor = matrix.getMatrix()[i][i];

        if (normalizationFactor == 0.0) {
            continue;
        }

        matrix.getMatrix()[i][i] /= normalizationFactor;
        vec.getVector()[i] /= normalizationFactor;
    }

//    cout << matrix << vec;
}

/*
 * Check if the given system of linear equations has at least one solution
 */
bool hasSolution(Matrix& matrix, ColumnVector& freeCoefficientsVector) {
    for (int i = 0; i < matrix.getRow(); i++) {
        bool areAllCoefficientsZero = true;
        for (int j = 0; j < matrix.getColumn(); j++) {
            if (fabs(matrix.getMatrix()[i][j]) > pow(10, -10)) {
                areAllCoefficientsZero = false;
            }
        }

        if (areAllCoefficientsZero && fabs(freeCoefficientsVector.getVector()[i]) > pow(10, -10)) {
            return false;
        }
    }

    return true;
}

/*
 * Check if the given system of linear equations has infinite number of solutions
 */
bool hasInfSolutions(Matrix& matrix, ColumnVector& freeCoefficientsVector) {
    for (int i = 0; i < matrix.getRow(); i++) {
        bool areAllCoefficientsZero = true;
        for (int j = 0; j < matrix.getColumn(); j++) {
            if (fabs(matrix.getMatrix()[i][j]) > pow(10, -10)) {
                areAllCoefficientsZero = false;
                break;
            }
        }

        if (areAllCoefficientsZero && fabs(freeCoefficientsVector.getVector()[i]) <= pow(10, -10)) {
            return true;
        }
    }

    return false;
}

ColumnVector solveLinearSystem(Matrix& matrix, ColumnVector& freeCoefficientsVector) {
    int step = 0;

//    cout << "step #" << step << ":\n";
//    cout << matrix << freeCoefficientsVector;
    step++;

    Matrix tempMatrix = matrix;
    ColumnVector tempVec = freeCoefficientsVector;

    directWay(tempMatrix, tempVec, &step);
    wayBack(tempMatrix, tempVec, &step);
    normalisation(tempMatrix, tempVec);

    if(!hasSolution(tempMatrix, tempVec)) {
        cout << "NO" << endl;
        exit(0);
    }

    if (hasInfSolutions(tempMatrix, tempVec)) {
        cout << "INF" << endl;
        exit(0);
    }

    return tempVec;
}

// Addition to the matrix overloaded operation
Matrix operator* (Matrix const& M1, ColumnVector const& V1) {
    if (M1.getColumn() != V1.getDimension()) {
        cout << errorMessage << endl;
        return {0, 0};
    }

    Matrix result(M1.getRow(), 1);

    for (int i = 0; i < result.getRow(); i++) {
        for (int j = 0; j < result.getColumn(); j++) {
            result.getMatrix()[i][j] = 0;
            for (int k = 0; k < M1.getColumn(); k++) {
                result.getMatrix()[i][j] += M1.getMatrix()[i][k] * V1.getVector()[k];
            }
        }
    }

    return result;
}

#ifdef WIN64
#define GNUPLOT_NAME "D:\\Studying\\gnuplot\\bin\\gnuplot -persist"
#else
#define GNUPLOT_NAME "gnuplot -persist"
#endif

int main() {

#ifdef WIN64
    FILE* plotter = _popen(GNUPLOT_NAME, "w");
#else
    FILE* plotter = popen(GNUPLOT_NAME, "w");
#endif

    if (plotter == nullptr) {
        return -1;
    }

    int datasetLength;
    vector<double> t;
    vector<double> b;
    int degreeOfPolynomial;

    // Parsing data
    cin >> datasetLength;
    for (int i = 0; i < datasetLength; i++) {
        double t1;
        double b1;

        cin >> t1 >> b1;

        t.push_back(t1);
        b.push_back(b1);
    }
    cin >> degreeOfPolynomial;

    // Creating vector of free coefficients
    ColumnVector vector(datasetLength);
    for (int i = 0; i < datasetLength; i++) {
        vector.getVector()[i] = b[i];
    }

    // Creating matrix
    int rowsCount = datasetLength;
    int columnsCount = degreeOfPolynomial + 1;
    Matrix augmentedMatrix(rowsCount, columnsCount);
    for (int i = 0; i < rowsCount; i++) {
        for (int j = 0; j < columnsCount; j++) {
            augmentedMatrix.getMatrix()[i][j] = pow(t[i], j);
        }
    }

    // Print Augmented matrix itself
    cout << "A:" << endl << augmentedMatrix;

    // Create and print converted to squared augmented matrix
    Matrix convertedToSquared = augmentedMatrix.transpose() * augmentedMatrix;
    cout << "A_T*A:" << endl << convertedToSquared;

    // Create and print inverse matrix of squared augmented matrix
    try {
        if (fabs(findDeterminant(convertedToSquared)) < pow(10, -10)) throw ZeroDet();

        getInverseMatrix(convertedToSquared);
    } catch (ZeroDet& e) {
        cout << e.what() << endl;
    }
    Matrix inverseMatrix = getInverseMatrix(convertedToSquared);
    cout << "(A_T*A)^-1:" << endl << inverseMatrix;

    // Create and print transposed matrix multiplied by free coefficients vector
    Matrix temporary = augmentedMatrix.transpose() * vector;
    cout << "A_T*b:" << endl << temporary;

    // The final answer
    Matrix vec = inverseMatrix * temporary;
    cout << "x~:" << endl << vec << endl;

    // Formatting plot
    fprintf(plotter, "%s\n", "set border linewidth 1.5\n"
                             "set style line 1 linecolor rgb '#0060ad' linetype 1 linewidth 2\n"
                             "set style line 2 linecolor rgb '#dd181f' linetype 1 linewidth 2");

    // Formatting legend
    fprintf(plotter, "%s\n", "set key at -50,70\n"
                             "set xlabel 'x'\n"
                             "set ylabel 'y'\n"
                             "set xrange [-100:100]\n"
                             "set yrange [-75:75]\n"
                             "set xtics 25\n"
                             "set ytics 25\n"
                             "set tics scale 0.75");

    // Plotting fit and raw data data
    fprintf(plotter, "%s", "f(x) = ");
    fprintf(plotter, "%f", vec.getMatrix()[0][0]);
    for (int i = 1; i < degreeOfPolynomial + 1; i++) {
        fprintf(plotter, "%s", " + ");
        fprintf(plotter, "%f", vec.getMatrix()[i][0]);
        fprintf(plotter, "%s", " * x**");
        fprintf(plotter, "%d", i);
    }
    fprintf(plotter, "%s\n", "");

    fprintf(plotter, "%s\n", "plot '-' title 'rawData' with lines linestyle 1, [x=-100:100] f(x) title 'fittedData' with "
                             "lines linestyle 2");
    for (int i = 0; i < datasetLength; i++) {
        fprintf(plotter, "%f%f\n", t[i], b[i]);
    }
    fprintf(plotter, "%c\n", 'e');


#ifdef WIN64
    _pclose(plotter);
#else
    pclose(plotter);
#endif

    return 0;
}
