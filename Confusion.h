/* 
 * File:   Confusion.h
 * Author: boyko_mihail
 *
 * Created on 2 ноября 2019 г., 23:02
 */

#ifndef CONFUSION_H
#define	CONFUSION_H

#include <eigen3/Eigen/Core>
#include <iostream>
#include <string>
#include <vector>
#include <tgmath.h> 
#include <sstream>  
#include <algorithm>
#include <array>

using namespace std;
using namespace Eigen;

class Confusion {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixEi;

public:
    int _classes;
    int _samples;
    double _c;
    MatrixEi _per;
    MatrixEi _cm;

    Confusion(MatrixEi targets, MatrixEi outputs) {
        confusion(targets, outputs);
    }

    void confusion(MatrixEi targets, MatrixEi outputs) {

        int numClasses = targets.rows();
        cout << " numClasses = " << numClasses << endl;
        if (numClasses == 1) {
            cout << "Number of classes must be greater than 1." << endl;
            return;
        }

        int numSamples = targets.row(0).cols();

        _classes = numClasses;
        _samples = numSamples;

        for (int col = 0; col < numSamples; col++) {
            double max = outputs.coeff(0, col);
            int ind = 0;

            for (int row = 1; row < numClasses; row++) {
                if (outputs.coeff(row, col) > max) {
                    max = outputs.coeff(row, col);
                    ind = row;
                }
                outputs.coeffRef(row, col) = 0.0;
            }
            outputs.coeffRef(0, col) = 0.0;
            outputs.coeffRef(ind, col) = 1.0;
        }

        //Confusion value
        int count = 0;
        for (int col = 0; col < numSamples; col++) {
            for (int row = 0; row < numClasses; row++) {
                if (targets.coeff(row, col) != outputs.coeff(row, col))
                    count++;
            }
        }
        double c = (double) count / (double) (2 * numSamples);

        // Confusion matrix
        MatrixEi cm;
        cm.setZero(numClasses, numClasses);

        VectorXd i(numSamples);
        VectorXd j(numSamples);

        for (int col = 0; col < numSamples; col++) {
            for (int row = 0; row < numClasses; row++) {
                if (targets.coeff(row, col) == 1.0) {
                    i[col] = row;
                    break;
                }
            }
        }

        for (int col = 0; col < numSamples; col++) {
            for (int row = 0; row < numClasses; row++) {
                if (outputs.coeff(row, col) == 1.0) {
                    j[col] = row;
                    break;
                }
            }
        }

        for (int col = 0; col < numSamples; col++) {
            cm.coeffRef(i[col], j[col]) = cm.coeff(i[col], j[col]) + 1;
        }

        //        // Indices
        //        vector<vector < string >> ind(numClasses, vector<string>(numClasses));
        //        for (int row = 0; row < numClasses; row++)
        //            for (int col = 0; col < numClasses; col++)
        //                ind[row][col] = "";
        //
        //
        //        for (int col = 0; col < numSamples; col++) {
        //            if (ind[i[col]][j[col]] == "")
        //                ind[i[col]][j[col]] = to_string(col);
        //            else
        //                ind[i[col]][j[col]] = ind[i[col]][j[col]].append(",").append(to_string(col));
        //        }

        // Percentages
        MatrixEi per;
        per.setZero(numClasses, 4);

        for (int row = 0; row < numClasses; row++) {

            auto yi = outputs.row(row);
            auto ti = targets.row(row);

            int a = 0, b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] != 1 && ti[col] == 1) a = a + 1;
                if (yi[col] != 1) b = b + 1;
            }
            per.coeffRef(row, 0) = (double) a / (double) b;


            a = 0;
            b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] == 1 && ti[col] != 1) a = a + 1;
                if (yi[col] == 1) b = b + 1;
            }
            per.coeffRef(row, 1) = (double) a / (double) b;



            a = 0;
            b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] == 1 && ti[col] == 1) a = a + 1;
                if (yi[col] == 1) b = b + 1;
            }
            per.coeffRef(row, 2) = (double) a / (double) b;


            a = 0;
            b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] != 1 && ti[col] != 1) a = a + 1;
                if (yi[col] != 1) b = b + 1;
            }
            per.coeffRef(row, 3) = (double) a / (double) b;

        }

        //NAN handling
        for (int row = 0; row < numClasses; row++) {
            for (int col = 0; col < 4; col++) {
                if (isnan(per.coeff(row, col)))
                    per.coeffRef(row, col) = 0;
            }
        }

        _c = c;
        _cm = cm;
        //  _ind = ind;
        _per = per;
    }

    string itos(int i) {
        stringstream s;
        s << i;
        return s.str();
    }

    float round(float valueToRound, int numberOfDecimalPlaces) {
        float multiplicationFactor = pow(10, numberOfDecimalPlaces);
        float interestedInZeroDPs = valueToRound * multiplicationFactor;
        return roundf(interestedInZeroDPs) / multiplicationFactor;
    }

    void printC() {
        cout << "\tConfusion value\n\t\tc = " << round(_c, 2) << endl;
    }

    void printCM() {
        cout << "\tConfusion Matrix" << endl;
        for (int row = 0; row < _classes; row++) {
            cout << "\t\t";
            for (int col = 0; col < _classes; col++) {
                cout << _cm.coeff(row, col) << " ";
            }
            cout << endl;
        }
    }

    //    void printInd() {
    //        cout << "\tIndices" << endl;
    //        for (int row = 0; row < _classes; row++) {
    //            for (int col = 0; col < _classes; col++) {
    //                cout << "\t\t[" << _ind[row][col] << "]";
    //            }
    //            cout << endl;
    //        }
    //    }

    void printPer() {
        cout << "\tPercentages" << endl;
        for (int row = 0; row < _classes; row++) {
            cout << "\t\t";
            for (int col = 0; col < 4; col++) {
                cout << round(_per.coeff(row, col), 2) << " ";
            }
            cout << endl;
        }
    }

    void print() {
        cout << "Confusion Results" << endl;
        cout << "=======================================" << endl;
        printC();
        printCM();
        //        printInd();
        printPer();
    }

    void print(vector<double> vec) {
        for (double d : vec) {
            cout << d << " ";
        }
        cout << endl;
    }

    void print(vector<vector<double>> vec) {
        for (int i = 0; i < vec.size(); ++i) {
            for (int j = 0; j < vec[0].size(); ++j) {
                cout << vec[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
};



#endif	/* CONFUSION_H */

