/* 
 * File:   ReLU.h
 * Author: boyko_mihail
 *
 * Created on 29 октября 2019 г., 17:48
 */

#ifndef RELU_H
#define	RELU_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"

class ReLU  : public Activation {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
    ReLU(){}
    // activation(z) = max(z, 0)

    void activate(const Matrix& Z, Matrix& A) {
        A.array() = Z.array().cwiseMax(Scalar(0));
    }

    // J = d_a / d_z = a > 0 ? 1 : 0
    // g = J * f = (a > 0 ? 1 : 0) .* f

    void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) {
        G.array() = (A.array() > Scalar(0)).select(F, Scalar(0));
    }
};


#endif	/* RELU_H */

