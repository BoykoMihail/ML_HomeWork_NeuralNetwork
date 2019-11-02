/* 
 * File:   Layer.h
 * Author: boyko_mihail
 *
 * Created on 28 октября 2019 г., 16:42
 */

#ifndef LAYER_H
#define	LAYER_H

#include <eigen3/Eigen/Core>
#include <vector>
#include "Optimizer.h"

class Layer {
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    const int m_in_size;
    const int m_out_size;

public:

    Layer(const int in_size, const int out_size) :
    m_in_size(in_size), m_out_size(out_size) {
    }

    virtual ~Layer() {
    }

    int in_size() const {
        return m_in_size;
    }

    int out_size() const {
        return m_out_size;
    }

    virtual void init(const Scalar& mu, const Scalar& sigma, RNG& rng) = 0;

    virtual void forward(const Matrix& prev_layer_data) = 0;


    virtual const Matrix& output() const = 0;

    virtual void backprop(const Matrix& prev_layer_data,
            const Matrix& next_layer_data) = 0;

    virtual const Matrix& backprop_data() const = 0;

    virtual void update(Optimizer& opt) = 0;

    virtual std::vector<Scalar> get_parameters() const = 0;

    virtual void set_parameters(const std::vector<Scalar>& param) {
    };

    virtual std::vector<Scalar> get_derivatives() const = 0;
};




#endif	/* LAYER_H */

