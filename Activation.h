/* 
 * File:   Activation.h
 * Author: boyko_mihail
 *
 * Created on 6 ноября 2019 г., 18:08
 */

#ifndef ACTIVATION_H
#define	ACTIVATION_H

#include <eigen3/Eigen/Core>
#include <string>

class Activation {
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::RowVectorXi IntegerVector;

public:
    Activation(){}

    virtual void activate(const Matrix& Z, Matrix& A) = 0;

    virtual void calculate_jacobian(const Matrix& Z, const Matrix& A,
            const Matrix& F, Matrix& G) = 0;

};


#endif	/* ACTIVATION_H */

