/* 
 * File:   VerboseCallback.h
 * Author: boyko_mihail
 *
 * Created on 30 октября 2019 г., 7:09
 */

#ifndef VERBOSECALLBACK_H
#define	VERBOSECALLBACK_H


#include <eigen3/Eigen/Core>
#include <iostream>
#include "Configuration.h"
#include "Callback.h"
#include "Network.h"

class VerboseCallback : public Callback {
public:

    void post_training_batch(const Network* net, const Matrix& x, const Matrix& y) {
        const Scalar loss = net->get_output()->loss();
        std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] Loss = "
                << loss << std::endl;
    }

    void post_training_batch(const Network* net, const Matrix& x,
            const IntegerVector& y) {
        Scalar loss = net->get_output()->loss();
        std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] Loss = "
                << loss << std::endl;
    }
};



#endif	/* VERBOSECALLBACK_H */

