/* 
 * File:   BinaryClassEntropy.h
 * Author: boyko_mihail
 *
 * Created on 1 ноября 2019 г., 21:24
 */

#include <eigen3/Eigen/Core>
#include <stdexcept>
#include "Configuration.h"

#ifndef BINARYCLASSENTROPY_H
#define	BINARYCLASSENTROPY_H

class BinaryClassEntropy : public Output {
private:
    Matrix m_din; // Derivative of the input.
public:

    void check_target_data(const Matrix& target) {
        const int nelem = target.size();
        const Scalar* target_data = target.data();

        for (int i = 0; i < nelem; i++) {
            if ((target_data[i] != Scalar(0)) && (target_data[i] != Scalar(1))) {
                throw std::invalid_argument("[class BinaryClassEntropy]: Target data should only contain zero or one");
            }
        }
    }

    void check_target_data(const IntegerVector& target) {
        const int nobs = target.size();

        for (int i = 0; i < nobs; i++) {
            if ((target[i] != 0) && (target[i] != 1)) {
                throw std::invalid_argument("[class BinaryClassEntropy]: Target data should only contain zero or one");
            }
        }
    }

    void evaluate(const Matrix& prev_layer_data, const Matrix& target) {
        const int nobs = prev_layer_data.cols();
        const int nvar = prev_layer_data.rows();

        if ((target.cols() != nobs) || (target.rows() != nvar)) {
            throw std::invalid_argument("[class BinaryClassEntropy]: Target data have incorrect dimension");
        }

        // L = -y * log(in) - (1 - y) * log(1 - in)
        // dL / d_in = -y / in + (1 - y) / (1 - in)
        m_din.resize(nvar, nobs);
        m_din.array() = (target.array() < Scalar(0.5)).select((Scalar(1) - prev_layer_data.array()).cwiseInverse(), -prev_layer_data.cwiseInverse());
    }

    void evaluate(const Matrix& prev_layer_data, const IntegerVector& target) {
        const int nvar = prev_layer_data.rows();

        if (nvar != 1) {
            throw std::invalid_argument("[class BinaryClassEntropy]: Only one response variable is allowed when class labels are used as target data");
        }

        const int nobs = prev_layer_data.cols();

        if (target.size() != nobs) {
            throw std::invalid_argument("[class BinaryClassEntropy]: Target data have incorrect dimension");
        }

        m_din.resize(1, nobs);
        m_din.array() = (target.array() == 0).select((Scalar(1) -
                prev_layer_data.array()).cwiseInverse(),
                -prev_layer_data.cwiseInverse());
    }

    const Matrix& backprop_data() const {
        return m_din;
    }

    Scalar loss() const {
        // L = -y * log(phat) - (1 - y) * log(1 - phat)
        // y = 0 => L = -log(1 - in)
        // y = 1 => L = -log(in)
        // m_din = 1/(1 - in) : y = 0
        // m_dim = -1/in : y = 1
        // L = log(abs(m_din)).sum()
        return m_din.array().abs().log().sum() / m_din.cols();
    }
};

#endif	/* BINARYCLASSENTROPY_H */

