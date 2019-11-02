/* 
 * File:   FullyConnected.h
 * Author: boyko_mihail
 *
 * Created on 31 октября 2019 г., 16:47
 */

#ifndef FULLYCONNECTED_H
#define	FULLYCONNECTED_H

#include <eigen3/Eigen/Core>
#include <vector>
#include <stdexcept>
#include "Configuration.h"
#include "Layer.h"
#include "Randome.h"

template <typename Activation>
class FullyConnected : public Layer {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

    Matrix m_weight; // W
    Vector m_bias; // Bias
    Matrix m_dw; // Derivative of W
    Vector m_db; // Derivative of b
    Matrix m_z; // z = W' * in + b
    Matrix m_a; // a = act(z)
    Matrix m_din; // Derivative of the input.

public:

    FullyConnected(const int in_size, const int out_size) :
    Layer(in_size, out_size) {
    }

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng) {
        m_weight.resize(this->m_in_size, this->m_out_size);
        m_bias.resize(this->m_out_size);
        m_dw.resize(this->m_in_size, this->m_out_size);
        m_db.resize(this->m_out_size);
        utilites::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
        utilites::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
    }

    void forward(const Matrix& prev_layer_data) {
        const int nobj = prev_layer_data.cols();
        // z = W' * in + b
        m_z.resize(this->m_out_size, nobj);
        m_z.noalias() = m_weight.transpose() * prev_layer_data;
        m_z.colwise() += m_bias;
        // activation
        m_a.resize(this->m_out_size, nobj);
        
        Activation::activate(m_z, m_a);
    }

    const Matrix& output() const {
        return m_a;
    }

    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) {
        const int nobj = prev_layer_data.cols();
        Matrix& dLz = m_z;
        Activation::calculate_jacobian(m_z, m_a, next_layer_data, dLz);
        m_dw.noalias() = prev_layer_data * dLz.transpose() / nobj;
        m_db.noalias() = dLz.rowwise().mean();
        m_din.resize(this->m_in_size, nobj);
        m_din.noalias() = m_weight * dLz;
    }

    const Matrix& backprop_data() const {
        return m_din;
    }

    void update(Optimizer& opt) {
        ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec w(m_weight.data(), m_weight.size());
        AlignedMapVec b(m_bias.data(), m_bias.size());
        opt.update(dw, w);
        opt.update(db, b);
    }

    std::vector<Scalar> get_parameters() const {
        std::vector<Scalar> res(m_weight.size() + m_bias.size());
        std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin());
        std::copy(m_bias.data(), m_bias.data() + m_bias.size(),
                res.begin() + m_weight.size());
        return res;
    }

    void set_parameters(const std::vector<Scalar>& param) {
        if (static_cast<int> (param.size()) != m_weight.size() + m_bias.size()) {
            throw std::invalid_argument("Parameter size does not match");
        }

        std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data());
        std::copy(param.begin() + m_weight.size(), param.end(), m_bias.data());
    }

    std::vector<Scalar> get_derivatives() const {
        std::vector<Scalar> res(m_dw.size() + m_db.size());
        std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
        std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_dw.size());
        return res;
    }
};

#endif	/* FULLYCONNECTED_H */

