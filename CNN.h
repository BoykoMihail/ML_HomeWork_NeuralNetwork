/* 
 * File:   CNN.h
 * Author: boyko_mihail
 *
 * Created on 4 ноября 2019 г., 20:38
 */

#ifndef CNN_H
#define	CNN_H

#include <eigen3/Eigen/Core>
#include <vector>
#include <stdexcept>
#include "Configuration.h"
#include "Layer.h"
#include "Randome.h"

template <typename Activation>
class CNN : public Layer {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Matrix::ConstAlignedMapType ConstAlignedMapMat;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;


    const ConvDims m_dim; // Various dimensions of convolution

    Matrix m_filter_data; // Filter parameters. Total length is
    // (in_channels x out_channels x filter_rows x filter_cols)
    // See Utils/Convolution.h for its layout

    Matrix m_df_data; // Derivative of filters, same dimension as m_filter_data

    Vector m_bias; // Bias term for the output channels, out_channels x 1. (One bias term per channel)
    Vector m_db; // Derivative of bias, same dimension as m_bias

    Matrix m_z; // Linear term, z = conv(in, w) + b. Each column is an observation
    Matrix m_a; // Output of this layer, a = act(z)
    Matrix m_din; // Derivative of the input of this layer
    // Note that input of this layer is also the output of previous layer

    Matrix Rot180(Matrix& input) {

        Matrix rot90 = input.transpose().colwise().reverse();
        return rot90.transpose().colwise().reverse();

    }

    Matrix conv2d(const Matrix& I, const Matrix &kernel) {
        Matrix O = Matrix::Zero(I.rows(), I.cols());


        typedef typename Matrix::Scalar Scalar;
        typedef typename Matrix::Scalar Scalar2;

        int col = 0, row = 0;
        int KSizeX = kernel.rows();
        int KSizeY = kernel.cols();

        int limitRow = I.rows() - KSizeX;
        int limitCol = I.cols() - KSizeY;

        Matrix block;
        Scalar normalization = kernel.sum();
        if (normalization < 1E-6) {
            normalization = 1;
        }
        for (row = KSizeX; row < limitRow; row++) {

            for (col = KSizeY; col < limitCol; col++) {
                
                cout<<"conv2d "<<col<<" "<<row<<endl;
                Scalar b = (static_cast<Matrix> (I.block(row, col, KSizeX, KSizeY)).cwiseProduct(kernel)).sum();
                O.coeffRef(row, col) = b;
            }
        }

        return O / normalization;
    }

public:

    CNN(const int in_width, const int in_height,
            const int in_channels, const int out_channels,
            const int window_width, const int window_height) :
    Layer(in_width * in_height * in_channels,
    (in_width - window_width + 1) * (in_height - window_height + 1) * out_channels),
    m_dim(in_channels, out_channels, in_height, in_width, window_height,
    window_width) {
    }

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng) {
        // Set data dimension
        const int filter_data_size = m_dim.in_channels * m_dim.out_channels *
                m_dim.filter_rows * m_dim.filter_cols;
        m_filter_data.resize(m_dim.filter_rows , m_dim.filter_cols);
        m_df_data.resize(m_dim.filter_rows , m_dim.filter_cols);
        // Random initialization of filter parameters
        utilites::set_normal_random(m_filter_data.data(), m_filter_data.size(), rng, mu,
                sigma);
        // Bias term
        m_bias.resize(m_dim.out_channels);
        m_db.resize(m_dim.out_channels);
        utilites::set_normal_random(m_bias.data(), m_dim.out_channels, rng, mu, sigma);
    }

    void forward(const Matrix& prev_layer_data) {
        // Each column is an observation
        const int nobs = prev_layer_data.cols();
        // Linear term, z = conv(in, w) + b
        m_z.resize(this->m_out_size, nobs);
        // Convolution
        m_z = conv2d(prev_layer_data, m_filter_data);
        
        //internal::convolve_valid(m_dim, prev_layer_data.data(), true, nobs,
        //                m_filter_data.data(), m_z.data()
        //                );
        // Add bias terms
        // Each column of m_z contains m_dim.out_channels channels, and each channel has
        // m_dim.conv_rows * m_dim.conv_cols elements
        int channel_start_row = 0;
        const int channel_nelem = m_dim.conv_rows * m_dim.conv_cols;

        for (int i = 0; i < m_dim.out_channels; i++, channel_start_row += channel_nelem) {
            m_z.block(channel_start_row, 0, channel_nelem, nobs).array() += m_bias[i];
        }

        // Apply activation function
        m_a.resize(this->m_out_size, nobs);
        Activation::activate(m_z, m_a);
    }

    const Matrix& output() const {
        return m_a;
    }

    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) {

        const int nobs = prev_layer_data.cols();

        Matrix& dLz = m_z;
        Activation::calculate_jacobian(m_z, m_a, next_layer_data, dLz);

        Matrix m_dimn = conv2d(next_layer_data, Rot180(m_filter_data));
        m_din.noalias() = m_dimn*dLz;

        m_df_data = conv2d(Rot180(m_din), prev_layer_data);

        // Derivative for bias
        // Aggregate d(L) / d(z) in each output channel
        ConstAlignedMapMat dLz_by_channel(dLz.data(), m_dim.conv_rows * m_dim.conv_cols,
                m_dim.out_channels * nobs);
        Vector dLb = dLz_by_channel.colwise().sum();
        // Average over observations
        ConstAlignedMapMat dLb_by_obs(dLb.data(), m_dim.out_channels, nobs);
        m_db.noalias() = dLb_by_obs.rowwise().mean();


    }

    const Matrix& backprop_data() const {
        return m_din;
    }

    void update(Optimizer& opt) {
        ConstAlignedMapVec dw(m_df_data.data(), m_df_data.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec w(m_filter_data.data(), m_filter_data.size());
        AlignedMapVec b(m_bias.data(), m_bias.size());
        opt.update(dw, w);
        opt.update(db, b);
    }

    std::vector<Scalar> get_parameters() const {
        std::vector<Scalar> res(m_filter_data.size() + m_bias.size());
        // Copy the data of filters and bias to a long vector
        std::copy(m_filter_data.data(), m_filter_data.data() + m_filter_data.size(),
                res.begin());
        std::copy(m_bias.data(), m_bias.data() + m_bias.size(),
                res.begin() + m_filter_data.size());
        return res;
    }

    void set_parameters(const std::vector<Scalar>& param) {
        if (static_cast<int> (param.size()) != m_filter_data.size() + m_bias.size()) {
            throw std::invalid_argument("Parameter size does not match");
        }

        std::copy(param.begin(), param.begin() + m_filter_data.size(),
                m_filter_data.data());
        std::copy(param.begin() + m_filter_data.size(), param.end(), m_bias.data());
    }

    std::vector<Scalar> get_derivatives() const {
        std::vector<Scalar> res(m_df_data.size() + m_db.size());
        // Copy the data of filters and bias to a long vector
        std::copy(m_df_data.data(), m_df_data.data() + m_df_data.size(), res.begin());
        std::copy(m_db.data(), m_db.data() + m_db.size(),
                res.begin() + m_df_data.size());
        return res;
    }
};


#endif	/* CNN_H */

