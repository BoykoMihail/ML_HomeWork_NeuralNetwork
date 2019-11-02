/* 
 * File:   Callback.h
 * Author: boyko_mihail
 *
 * Created on 30 октября 2019 г., 11:07
 */

#ifndef CALLBACK_H
#define	CALLBACK_H

#include <eigen3/Eigen/Core>
#include "Network.h"

class Network;

class Callback {
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::RowVectorXi IntegerVector;

public:

    int m_nbatch; // Count of batch
    int m_batch_id; // current batch number
    int m_nepoch; // Count of epoches
    int m_epoch_id; // current epoch number

    Callback() :
    m_nbatch(0), m_batch_id(0), m_nepoch(0), m_epoch_id(0) {
    }

    virtual ~Callback() {
    }

    virtual void pre_training_batch(const Network* net, const Matrix& x,
            const Matrix& y) {
    }

    virtual void pre_training_batch(const Network* net, const Matrix& x,
            const IntegerVector& y) {
    }

    virtual void post_training_batch(const Network* net, const Matrix& x,
            const Matrix& y) {
    }

    virtual void post_training_batch(const Network* net, const Matrix& x,
            const IntegerVector& y) {
    }
};


#endif	/* CALLBACK_H */

