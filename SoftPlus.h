/* 
 * File:   SoftPlus.h
 * Author: boyko_mihail
 *
 * Created on 30 октября 2019 г., 19:47
 */

#ifndef SOFTPLUS_H
#define	SOFTPLUS_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"

class SoftPlus {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
    // activation(z) = log (1 + exp(-z))

    static inline void activate(const Matrix& Z, Matrix& A) {
        A.array() = (Scalar(1) + (-Z.array()).exp()).log();
    }

    // J = d_a / d_z = 1 / 1 + exp(-z))
    // g = J * f = z / 1 + exp(-z))

    static inline void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) {
        G.array() = Z.array() / (Scalar(1) + (-Z.array()).exp());
    }
};


#endif	/* SOFTPLUS_H */

