/*
 * File:   DropOut.h
 * Author: boyko_mihail
 *
 * Created on 31 октября 2019 г., 22:25
 */

#ifndef DROPOUT_H
#define	DROPOUT_H


#include <eigen3/Eigen/Core>
#include <vector>
#include <stdexcept>
#include "Configuration.h"
#include "Layer.h"
#include "Randome.h"

class DropOut : public Layer {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    Matrix dropout_mask;
    float dropout_ratio = 1.0f;

    Matrix m_a;
    Matrix m_din;

public:

    DropOut(const int in_size, const int out_size) :
    Layer(in_size, out_size) {
    }

    void set_dropout_ratio(float dropout_ratio) {
        this-> dropout_ratio = dropout_ratio;
    }

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng) {
    }

    void forward(const Matrix& prev_layer_data) {
        dropout_mask = utilites::binomial(prev_layer_data.rows(), prev_layer_data.cols(), dropout_ratio) / dropout_ratio;
        m_a = prev_layer_data.cwiseProduct(dropout_mask);
    }

    const Matrix& output() const {
        return m_a;
    }

    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) {
        m_din = next_layer_data.cwiseProduct(dropout_mask);
    }

    const Matrix& backprop_data() const {
        return m_din;
    }

    void update(Optimizer& opt) {
    }

    std::vector<Scalar> get_parameters() const {
        std::vector<Scalar> res(0);
        return res;
    }

    void set_parameters(const std::vector<Scalar>& param) {
    }

    std::vector<Scalar> get_derivatives() const {
        std::vector<Scalar> res(0);
        return res;
    }
};

#endif	/* DROPOUT_H */

