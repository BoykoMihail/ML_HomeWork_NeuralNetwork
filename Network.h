/* 
 * File:   Network.h
 * Author: boyko_mihail
 *
 * Created on 28 октября 2019 г., 20:56
 */

#ifndef NETWORK_H
#define	NETWORK_H

#include <eigen3/Eigen/Core>
#include <vector>
#include <stdexcept>
#include "Configuration.h"
#include "RNG.h"
#include "Layer.h"
#include "Output.h"
#include "Callback.h"
#include "Randome.h"
#include <iostream>
#include "DropOut.h"

using namespace std;

class Network {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::RowVectorXi IntegerVector;

    RNG m_default_rng;
    RNG& m_rng;
    std::vector<Layer*> m_layers;
    Output* m_output;
    Callback m_default_callback;
    Callback*
    m_callback;

    void check_unit_sizes() const {
        const int nlayer = num_layers();

        if (nlayer <= 1) {
            return;
        }

        for (int i = 1; i < nlayer; i++) {
            if ( m_layers[i]->getNameOfLayer() != "DropOut" && m_layers[i - 1]->getNameOfLayer() != "DropOut" && m_layers[i]->in_size() != m_layers[i - 1]->out_size()) {
                throw std::invalid_argument("Unit sizes do not match");
            }
        }
    }

    void forward(const Matrix& input) {
        const int nlayer = num_layers();

        if (nlayer <= 0) {
            return;
        }
//
//         for (int i = 1; i < nlayer; i++) {
//            if ( m_layers[i]->getNameOfLayer() != "DropOut" && m_layers[i - 1]->getNameOfLayer() != "DropOut" && m_layers[i]->in_size() != m_layers[i - 1]->out_size()) {
//                throw std::invalid_argument("Unit sizes do not match");
//            }
//        }
        
        if (input.rows() != m_layers[0]->in_size()) {
            throw std::invalid_argument("Input data have incorrect dimension");
        }

        m_layers[0]->forward(input);

        for (int i = 1; i < nlayer; i++) {
            m_layers[i]->forward(m_layers[i - 1]->output());
        }
    }

    template <typename TargetType>
    void backprop(const Matrix& input, const TargetType& target) {
        const int nlayer = num_layers();

        if (nlayer <= 0) {
            return;
        }

        Layer* first_layer = m_layers[0];
        Layer* last_layer = m_layers[nlayer - 1];
        m_output->check_target_data((Matrix) target);
        m_output->evaluate(last_layer->output(), (Matrix) target);

        if (nlayer == 1) {
            first_layer->backprop(input, m_output->backprop_data());
            return;
        }

        last_layer->backprop(m_layers[nlayer - 2]->output(), m_output->backprop_data());

        for (int i = nlayer - 2; i > 0; i--) {
            m_layers[i]->backprop(m_layers[i - 1]->output(),
                    m_layers[i + 1]->backprop_data());
        }

        first_layer->backprop(input, m_layers[1]->backprop_data());
    }

    void update(Optimizer& opt) {
        const int nlayer = num_layers();

        if (nlayer <= 0) {
            return;
        }

        for (int i = 0; i < nlayer; i++) {
            m_layers[i]->update(opt);
        }
    }

public:

    Network() :
    m_default_rng(1),
    m_rng(m_default_rng),
    m_output(NULL),
    m_default_callback(),
    m_callback(&m_default_callback) {
    }

    Network(RNG& rng) :
    m_default_rng(1),
    m_rng(rng),
    m_output(NULL),
    m_default_callback(),
    m_callback(&m_default_callback) {
    }

    ~Network() {
        const int nlayer = num_layers();

        for (int i = 0; i < nlayer; i++) {
            delete m_layers[i];
        }

        if (m_output) {
            delete m_output;
        }
    }

    void add_layer(Layer* layer) {
        m_layers.push_back(layer);
    }

    void set_output(Output* output) {
        if (m_output) {
            delete m_output;
        }

        m_output = output;
    }

    int num_layers() const {
        return m_layers.size();
    }

    std::vector<const Layer*> get_layers() const {
        const int nlayer = num_layers();
        std::vector<const Layer*> layers(nlayer);
        std::copy(m_layers.begin(), m_layers.end(), layers.begin());
        return layers;
    }

    const Output* get_output() const {
        return m_output;
    }

    void set_callback(Callback& callback) {
        m_callback = &callback;
    }

    void set_default_callback() {
        m_callback = &m_default_callback;
    }

    void init(const Scalar& mu = Scalar(0), const Scalar& sigma = Scalar(0.01),
            int seed = -1) {
        check_unit_sizes();

        if (seed > 0) {
            m_rng.seed(seed);
        }

        const int nlayer = num_layers();

        for (int i = 0; i < nlayer; i++) {
            m_layers[i]->init(mu, sigma, m_rng);
        }
    }

    std::vector< std::vector<Scalar>> get_parameters() const {
        const int nlayer = num_layers();
        std::vector< std::vector < Scalar>> res;
        res.reserve(nlayer);

        for (int i = 0; i < nlayer; i++) {
            res.push_back(m_layers[i]->get_parameters());
        }

        return res;
    }

    void set_parameters(const std::vector< std::vector<Scalar>>&param) {
        const int nlayer = num_layers();

        if (static_cast<int> (param.size()) != nlayer) {
            throw std::invalid_argument("Parameter size does not match");
        }

        for (int i = 0; i < nlayer; i++) {
            m_layers[i]->set_parameters(param[i]);
        }
    }

    std::vector< std::vector<Scalar>> get_derivatives() const {
        const int nlayer = num_layers();
        std::vector< std::vector < Scalar>> res;
        res.reserve(nlayer);

        for (int i = 0; i < nlayer; i++) {
            res.push_back(m_layers[i]->get_derivatives());
        }

        return res;
    }

    template <typename TargetType>
    void check_gradient(const Matrix& input, const TargetType& target, int npoints,
            int seed = -1) {
        if (seed > 0) {
            m_rng.seed(seed);
        }

        this->forward(input);
        this->backprop(input, target);
        std::vector< std::vector < Scalar>> param = this->get_parameters();
        std::vector< std::vector < Scalar>> deriv = this->get_derivatives();
        const Scalar eps = 1e-5;
        const int nlayer = deriv.size();

        for (int i = 0; i < npoints; i++) {
            // Randomly select a layer
            const int layer_id = int(m_rng.rand() * nlayer);
            // Randomly pick a parameter, note that some layers may have no parameters
            const int nparam = deriv[layer_id].size();

            if (nparam < 1) {
                continue;
            }

            const int param_id = int(m_rng.rand() * nparam);
            // Turbulate the parameter a little bit
            const Scalar old = param[layer_id][param_id];
            param[layer_id][param_id] -= eps;
            this->set_parameters(param);
            this->forward(input);
            this->backprop(input, target);
            const Scalar loss_pre = m_output->loss();
            param[layer_id][param_id] += eps * 2;
            this->set_parameters(param);
            this->forward(input);
            this->backprop(input, target);
            const Scalar loss_post = m_output->loss();
            const Scalar deriv_est = (loss_post - loss_pre) / eps / 2;
            std::cout << "[layer " << layer_id << ", param " << param_id <<
                    "] deriv = " << deriv[layer_id][param_id] << ", est = " << deriv_est <<
                    ", diff = " << deriv_est - deriv[layer_id][param_id] << std::endl;
            param[layer_id][param_id] = old;
        }

        // Restore original parameters
        this->set_parameters(param);
    }

    template <typename DerivedX, typename DerivedY>
    bool fit(Optimizer& opt, const Eigen::MatrixBase<DerivedX>& x,
            const Eigen::MatrixBase<DerivedY>& y,
            int batch_size, int epoch, int seed = -1) {
        typedef typename Eigen::MatrixBase<DerivedX>::PlainObject PlainObjectX;
        typedef typename Eigen::MatrixBase<DerivedY>::PlainObject PlainObjectY;
        typedef Eigen::Matrix<typename PlainObjectX::Scalar, PlainObjectX::RowsAtCompileTime, PlainObjectX::ColsAtCompileTime>
                XType;
        typedef Eigen::Matrix<typename PlainObjectY::Scalar, PlainObjectY::RowsAtCompileTime, PlainObjectY::ColsAtCompileTime>
                YType;
        const int nlayer = num_layers();

        if (nlayer <= 0) {
            return false;
        }

        opt.reset();

        if (seed > 0) {
            m_rng.seed(seed);
        }



        for (int k = 0; k < epoch; k++) {

            std::vector<XType> x_batches;
            std::vector<YType> y_batches;

            const int nbatch = utilites::create_shuffled_batches(x, y, batch_size, m_rng,
                    x_batches, y_batches);

            m_callback->m_nbatch = nbatch;
            m_callback->m_nepoch = epoch;


            m_callback->m_epoch_id = k;

            for (int i = 0; i < nbatch; i++) {

                m_callback->m_batch_id = i;
                m_callback->pre_training_batch(this, x_batches[i], (Matrix) y_batches[i]);
                
                this->forward(x_batches[i]);
                this->backprop(x_batches[i], y_batches[i]);
                this->update(opt);
                m_callback->post_training_batch(this, x_batches[i], (Matrix) y_batches[i]);

            }
        }

        return true;
    }

    Matrix predict(const Matrix& x) {
        const int nlayer = num_layers();

        if (nlayer <= 0) {
            return Matrix();
        }

        this->forward(x);
        return m_layers[nlayer - 1]->output();
    }
};

#endif	/* NETWORK_H */

