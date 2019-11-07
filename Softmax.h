/* 
 * File:   Softmax.h
 * Author: boyko_mihail
 *
 * Created on 30 октября 2019 г., 17:50
 */

#ifndef SOFTMAX_H
#define	SOFTMAX_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"

class Softmax : public Activation{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> RowArray;

public:
    Softmax(){}
    // a = activation(z) = softmax(z)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations

    void activate(const Matrix& Z, Matrix& A) {
        A.array() = (Z.rowwise() - Z.colwise().maxCoeff()).array().exp();
        RowArray colsums = A.colwise().sum();
        A.array().rowwise() /= colsums;
    }

    
    // J = d_a / d_z = diag(a) - a * a'
    // g = J * f = a .* f - a * (a' * f) = a .* (f - a'f)

    void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) {
        RowArray a_dot_f = A.cwiseProduct(F).colwise().sum();
        G.array() = A.array() * (F.array().rowwise() - a_dot_f);
    }
};



#endif	/* SOFTMAX_H */

