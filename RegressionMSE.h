/* 
 * File:   RegressionMSE.h
 * Author: boyko_mihail
 *
 * Created on 1 ноября 2019 г., 23:30
 */


#include <eigen3/Eigen/Core>
#include <stdexcept>
#include "Configuration.h"

#ifndef REGRESSIONMSE_H
#define	REGRESSIONMSE_H

class RegressionMSE : public Output {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    Matrix m_din; // Derivative of the input.

public:

    void evaluate(const Matrix& prev_layer_data, const Matrix& target) {
        const int nobs = prev_layer_data.cols();
        const int nvar = prev_layer_data.rows();

        if ((target.cols() != nobs) || (target.rows() != nvar)) {
            throw std::invalid_argument("[class RegressionMSE]: Target data have incorrect dimension");
        }

        // Calculate the derivative of the input
        // L = 0.5 * ||in - y||^2
        // d(L) / d(in) = in - y
        m_din.resize(nvar, nobs);
        m_din.noalias() = prev_layer_data - target;
    }

    const Matrix& backprop_data() const {
        return m_din;
    }

    Scalar loss() const {
        // L = 0.5 * ||in - y||^2
        return m_din.squaredNorm() / m_din.cols() * Scalar(0.5);
    }
};

#endif	/* REGRESSIONMSE_H */

