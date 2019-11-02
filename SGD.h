/* 
 * File:   SGD.h
 * Author: boyko_mihail
 *
 * Created on 1 ноября 2019 г., 20:37
 */

#ifndef SGD_H
#define	SGD_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"
#include "Optimizer.h"

class SGD : public Optimizer {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

public:
    Scalar m_lrate;
    Scalar m_decay;

    SGD() :
    m_lrate(Scalar(0.01)), m_decay(Scalar(0)) {
    }

    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) {
        vec.noalias() -= m_lrate * (dvec + m_decay * vec);
    }
};



#endif	/* SGD_H */

