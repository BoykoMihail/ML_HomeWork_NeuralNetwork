/* 
 * File:   LogSoftMax.h
 * Author: boyko_mihail
 *
 * Created on 31 октября 2019 г., 22:11
 */

#ifndef LOGSOFTMAX_H
#define	LOGSOFTMAX_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"
#include <cstdlib>
#include <iostream>

using namespace std;

class LogSoftMax : public Activation {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> RowArray;

public:
     LogSoftMax(){}
    // activation(z) = log(softmax(z))

    void activate(const Matrix& Z, Matrix& A) {
        A.array() = (Z.rowwise() - Z.colwise().maxCoeff()).array().exp();
        RowArray colsums = A.colwise().sum();
        A.array() =  (A.array().log().rowwise() -  (colsums).log()).array();
    }

    // J = d_a / d_z = z - sum(z*exp(z)) / sum (exp(z))
    // g = J * f = z .* f - sum(z.*a).*f

    void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) {

        auto col_temp = ((A.array() * (Z.rowwise() - Z.colwise().maxCoeff()).array()).colwise().sum()) ;
        auto colsums = col_temp.array() / A.colwise().sum().array();        
        G.array() = (((Z.rowwise() - Z.colwise().maxCoeff()).array().rowwise() - colsums).array() * F.array());
    }

};


#endif	/* LOGSOFTMAX_H */

