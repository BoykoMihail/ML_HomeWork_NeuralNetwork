/* 
 * File:   Output.h
 * Author: boyko_mihail
 *
 * Created on 31 октября 2019 г., 22:20
 */

#ifndef OUTPUT_H
#define	OUTPUT_H

#include <eigen3/Eigen/Core>
#include <stdexcept>
#include "Configuration.h"

class Output {
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::RowVectorXi IntegerVector;

public:

    virtual ~Output() {
    }

    virtual void check_target_data(const Matrix& target) {
    }

    virtual void check_target_data(const IntegerVector& target) {
    }

    virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;

    virtual void evaluate(const Matrix& prev_layer_data,
            const IntegerVector& target) {
    }
    virtual const Matrix& backprop_data() const = 0;

    virtual Scalar loss() const = 0;
};

#endif	/* OUTPUT_H */

