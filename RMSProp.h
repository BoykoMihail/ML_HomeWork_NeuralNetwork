/* 
 * File:   RMSProp.h
 * Author: boyko_mihail
 *
 * Created on 1 ноября 2019 г., 20:04
 */

#include <eigen3/Eigen/Core>
#include "Configuration.h"
#include "Optimizer.h"


#ifndef RMSPROP_H
#define	RMSPROP_H

class RMSProp : public Optimizer {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;


public:
    Scalar m_lrate;
    Scalar m_eps;
    Scalar m_decay;

    RMSProp() :
    m_lrate(Scalar(0.001)), m_eps(Scalar(1e-6)), m_decay(Scalar(0.9)) {
    }

    void reset() {
    }

    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) {
        Array grad_square;
        grad_square.resize(dvec.size());
        grad_square.setZero();
        grad_square = m_decay * grad_square + (Scalar(1) - m_decay) *
                dvec.array().square();
        vec.array() -= m_lrate * dvec.array() / (grad_square + m_eps).sqrt();
    }
};


#endif	/* RMSPROP_H */

