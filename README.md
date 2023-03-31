#  Linear-Algebra
This repository contains the implementation of Matrix / Column Vectors behavior for solving linear systems of equations through: Gauss elimination, Jacobi iterative methods, and Seidel iterative methods. The program implements Least Square Approximation, which gave an approximate value for coefficients of n-degree polynomial. 

## **cin >> inputFormat >> endl**;
--> cin >> ...;

Matrix and Column vectors can be instantiated using the size; Overloaded operators for input for both classes.
```
Example of matrix input:                                      Example of column vector input:

3                         // Number of rows                   3                               // vector's dimension

3                         // Number of columns                1                               // vector itself

1 2 3                     // Matrix itself                    2

4 5 6                                                         3

7 8 9
```

## **LSA input format**
LSA implementation need a several variables: datasetLength, array T, array B, degree of polynomial. You also should point to the directory of the gnuplot.exe file just before the 
```cpp
int main() {...}
``` 
function. If you done with this, in the end of the approximation, program start the plotter and show the raw data and fitted data by different lines.

For instance, the following input for the LSA:

```
3         // datasetLength
1 1       // t_{i} b_{i}
2 4
3 9
2         // degree of polynomial
```

## **cin << outputFormat << endl**;
--> cout << ...;

Matrix and Column vecors can be outputed through the ordinary output streams, for instance "cout << matrix;"
There are can be added a step by step output for every part of the Gauss Elimination, intermediate matrices and column vectors, etc <--> Find a certain commented lines and uncomment them.

## **LSA output format**
The program output in the console the most approximated value for the coefficients of current n-degree polynomial. If actions with the gnuplot.exe was done correctly, gnuplot.exe will start and show the graphs, one for the raw data, and one for the fitted data.

## **Possible actions with matrices and column vectors**
Overloaded all necessary operators for both classes: **summation, subtractions, multiplication**. 

A deeper understanding of possible operations with both classes can be obtained by skimming the program itself. All difficult parts in the code commented as well as possible. 

## **Addition information**

The repository contains several additional files, which are the following:

--> **_generator**.py_ -- program, which implements the simple generator for creating the set of points for the LSA, is written in Python.

--> **_repontOnLSA_** -- small article about how the LSA works, with the samples.

--> **_test1.png_** && **_test2.png_** -- samples of the plotter work, which are included in the _repontOnLSA_.

Several files, the names of which describe themselves better, You can implement by yourself every particular case using the **_main.cpp_**.



