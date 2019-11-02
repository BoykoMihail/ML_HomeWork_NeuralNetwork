/* 
 * File:   Sigmoid.h
 * Author: boyko_mihail
 *
 * Created on 29 октября 2019 г., 17:49
 */

#ifndef SIGMOID_H
#define	SIGMOID_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"

class Sigmoid {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
    // activation(z) = 1 / (1 + exp(-z))

    static inline void activate(const Matrix& Z, Matrix& A) {
        A.array() = Scalar(1) / (Scalar(1) + (-Z.array()).exp());
    }

    // J = d_a / d_z = a .* (1 - a)
    // g = J * f = a .* (1 - a) .* f

    static inline void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) {
        G.array() = A.array() * (Scalar(1) - A.array()) * F.array();
    }
};

#endif	/* SIGMOID_H */

