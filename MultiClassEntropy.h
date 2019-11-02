/* 
 * File:   MultiClassEntropy.h
 * Author: boyko_mihail
 *
 * Created on 1 ноября 2019 г., 22:51
 */

#ifndef MULTICLASSENTROPY_H
#define	MULTICLASSENTROPY_H


#include <eigen3/Eigen/Core>
#include <stdexcept>
#include "Configuration.h"

class MultiClassEntropy : public Output {
private:
    Matrix m_din; // Derivative of the input.
public:

    void check_target_data(const Matrix& target) {
        const int nobs = target.cols();
        const int nclass = target.rows();

        for (int i = 0; i < nobs; i++) {
            int one = 0;

            for (int j = 0; j < nclass; j++) {
                if (target(j, i) == Scalar(1)) {
                    one++;
                    continue;
                }

                if (target(j, i) != Scalar(0)) {
                    throw std::invalid_argument("[class MultiClassEntropy]: Target data should only contain zero or one");
                }
            }

            if (one != 1) {
                throw std::invalid_argument("[class MultiClassEntropy]: Each column of target data should only contain one \"1\"");
            }
        }
    }

    void check_target_data(const IntegerVector& target) {
        const int nobs = target.size();

        for (int i = 0; i < nobs; i++) {
            if (target[i] < 0) {
                throw std::invalid_argument("[class MultiClassEntropy]: Target data must be non-negative");
            }
        }
    }

    void evaluate(const Matrix& prev_layer_data, const Matrix& target) {
        const int nobs = prev_layer_data.cols();
        const int nclass = prev_layer_data.rows();

        if ((target.cols() != nobs) || (target.rows() != nclass)) {
            throw std::invalid_argument("[class MultiClassEntropy]: Target data have incorrect dimension");
        }

        // Calculate the derivative of the input
        // L = -sum(log(in) * y)
        // d(L) / d(in) = -y / in
        m_din.resize(nclass, nobs);

        //        cout<<endl<<endl<<" prev_layer_data = "<<prev_layer_data.col(0)<<endl<<endl;
        m_din.noalias() = -target.cwiseQuotient(prev_layer_data);
        //        cout<<"m_dim = "<<m_din.col(0)<<endl<<endl<<endl<<endl;
    }

    void evaluate(const Matrix& prev_layer_data, const IntegerVector& target) {
        const int nobs = prev_layer_data.cols();
        const int nclass = prev_layer_data.rows();

        if (target.size() != nobs) {
            throw std::invalid_argument("[class MultiClassEntropy]: Target data have incorrect dimension");
        }

        // Calculate the derivative of the input
        // L = -log(in[y])
        // d(L) / d(in) = [0, 0, ..., -1/in[y], 0, ..., 0]
        m_din.resize(nclass, nobs);
        m_din.setZero();

        for (int i = 0; i < nobs; i++) {
            m_din(target[i], i) = -Scalar(1) / prev_layer_data(target[i], i);
        }
    }

    const Matrix& backprop_data() const {
        return m_din;
    }

    Scalar loss() const {
        // L = -sum(log(in) * y)
        // d(L) / d(in) = -y / in
        // m_din = 0 : y = 0, and -1/phat : y = 1
        Scalar res = Scalar(0);
        const int nelem = m_din.size();
        const Scalar* din_data = m_din.data();

        for (int i = 0; i < nelem; i++) {
            if (din_data[i] < Scalar(0)) {
                res += abs(std::log(-din_data[i]));
            } 
        }

        return res / m_din.cols();
    }
};


#endif	/* MULTICLASSENTROPY_H */

