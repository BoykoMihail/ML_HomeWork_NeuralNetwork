/* 
 * File:   Optimizer.h
 * Author: boyko_mihail
 *
 * Created on 29 октября 2019 г., 23:53
 */

#ifndef OPTIMIZER_H
#define	OPTIMIZER_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"

class Optimizer {
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

public:

    virtual ~Optimizer() {
    }

    virtual void reset() {
    };

    virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};



#endif	/* OPTIMIZER_H */

