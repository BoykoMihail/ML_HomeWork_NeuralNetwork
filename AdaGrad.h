/* 
 * File:   AdaGrad.h
 * Author: boyko_mihail
 *
 * Created on 1 ноября 2019 г., 17:24
 */

#ifndef ADAGRAD_H
#define	ADAGRAD_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"
#include "Optimizer.h"

class AdaGrad : public Optimizer {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

public:
    Scalar m_lrate;
    Scalar m_eps;

    AdaGrad() :
    m_lrate(Scalar(0.01)), m_eps(Scalar(1e-7)) {
    }

    void reset() {
    }

    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) {
        Array grad_square;
        grad_square.resize(dvec.size());
        grad_square.setZero();
        grad_square += dvec.array().square();
        vec.array() -= m_lrate * dvec.array() / (grad_square.sqrt() + m_eps);
    }
};

#endif	/* ADAGRAD_H */

