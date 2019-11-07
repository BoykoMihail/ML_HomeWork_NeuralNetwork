/* 
 * File:   LeakyRelu.h
 * Author: boyko_mihail
 *
 * Created on 29 октября 2019 г., 21:41
 */

#ifndef LEAKYRELU_H
#define	LEAKYRELU_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"

class LeakyRelu : public Activation{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    double incline;
public:
    
    LeakyRelu(double inc = 0.01){
        incline = inc;
    }
    // activation(z) = max(z, 0)

    void activate(const Matrix& Z, Matrix& A) {
        A.array() = Z.array().cwiseMax(Scalar(incline) * Z.array());
    }

    // J = d_a / d_z = a > 0 ? 1 : 0.01
    // g = J * f = (a > 0 ? 1 : 0.01) .* f

    void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) {
        G.array() = (A.array() > Scalar(0)).select(F, Scalar(incline) * F.array());
    }
};



#endif	/* LEAKYRELU_H */

