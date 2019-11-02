/* 
 * File:   Identity.h
 * Author: boyko_mihail
 *
 * Created on 29 октября 2019 г., 17:47
 */

#ifndef IDENTITY_H
#define	IDENTITY_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"

class Identity {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:

    static inline void activate(const Matrix& Z, Matrix& A) {
        A.noalias() = Z;
    }

    // J = d_a / d_z = I
    // g = J * f = f

    static inline void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) {
        G.noalias() = F;
    }
};

#endif	/* IDENTITY_H */

