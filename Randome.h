/* 
 * File:   Randome.h
 * Author: boyko_mihail
 *
 * Created on 30 октября 2019 г., 11:34
 */

#ifndef RANDOME_H
#define	RANDOME_H

#include <eigen3/Eigen/Core>
#include "Configuration.h"
#include "RNG.h"
#include <vector>
#include <random>
#include <iostream>

using namespace std;

struct ConvDims {
    // Input parameters
    const int in_channels;
    const int out_channels;
    const int channel_rows;
    const int channel_cols;
    const int filter_rows;
    const int filter_cols;
    // Image dimension -- one observation with all channels
    const int img_rows;
    const int img_cols;
    // Dimension of the convolution result for each output channel
    const int conv_rows;
    const int conv_cols;

    ConvDims(
            const int in_channels_, const int out_channels_,
            const int channel_rows_, const int channel_cols_,
            const int filter_rows_, const int filter_cols_
            ) :
    in_channels(in_channels_), out_channels(out_channels_),
    channel_rows(channel_rows_), channel_cols(channel_cols_),
    filter_rows(filter_rows_), filter_cols(filter_cols_),
    img_rows(channel_rows_), img_cols(in_channels_ * channel_cols_),
    conv_rows(channel_rows_ - filter_rows_ + 1),
    conv_cols(channel_cols_ - filter_cols_ + 1) {
    }
};

namespace utilites {

    inline void shuffle(int* arr, const int n, RNG& rng) {
        for (int i = n - 1; i > 0; i--) {
            const int j = int(rng.rand() * (i + 1));
            const int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    template <typename DerivedX, typename DerivedY, typename XType, typename YType>
    int create_shuffled_batches(
            const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y,
            int batch_size, RNG& rng,
            std::vector<XType>& x_batches, std::vector<YType>& y_batches
            ) {
        
        
        
        const int nobs = x.cols();
        const int dimx = x.rows();
        const int dimy = y.rows();

       
        if (y.cols() != nobs) {
            throw std::invalid_argument("Input X and Y have different number of observations");
        }
        
        Eigen::VectorXi id = Eigen::VectorXi::LinSpaced(nobs, 0, nobs - 1);
         
        shuffle(id.data(), id.size(), rng);


        if (batch_size > nobs) {
            batch_size = nobs;
        }

        const int nbatch = (nobs - 1) / batch_size + 1;
        const int last_batch_size = nobs - (nbatch - 1) * batch_size;
        x_batches.clear();
        y_batches.clear();
        x_batches.reserve(nbatch);
        y_batches.reserve(nbatch);

        for (int i = 0; i < nbatch; i++) {
            const int bsize = (i == nbatch - 1) ? last_batch_size : batch_size;
            x_batches.push_back(XType(dimx, bsize));
            y_batches.push_back(YType(dimy, bsize));
            const int offset = i * batch_size;

            for (int j = 0; j < bsize; j++) {
                x_batches[i].col(j).noalias() = x.col(id[offset + j]);
                y_batches[i].col(j).noalias() = y.col(id[offset + j]);
            }
        }

        return nbatch;
    }

    inline void set_normal_random(Scalar* arr, const int n, RNG& rng,
            const Scalar& mu = Scalar(0),
            const Scalar& sigma = Scalar(1)) {
        const double two_pi = 6.283185307179586476925286766559;

        for (int i = 0; i < n - 1; i += 2) {
            const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
            const double t2 = two_pi * rng.rand();
            arr[i] = t1 * std::cos(t2) + mu;
            arr[i + 1] = t1 * std::sin(t2) + mu;
        }

        if (n % 2 == 1) {
            const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
            const double t2 = two_pi * rng.rand();
            arr[n - 1] = t1 * std::cos(t2) + mu;
        }
    }

    void shuffleMatrixPair(Eigen::MatrixXd& mat1, Eigen::MatrixXd& mat2) {
        if (mat1.rows() != mat2.rows() || mat2.cols() != mat2.cols()) {
            std::cerr << "Not possible to shuffle, dimension problem!" << std::endl;
        }
        int half = static_cast<int> (mat1.rows() * 0.5);
        for (auto i = 0; i < half; i++) {
            int swap_index = rand() % mat1.rows();
            mat1.row(i).swap(mat1.row(swap_index));
            mat2.row(i).swap(mat2.row(swap_index));
        }
    }

    void splitMatrixPair(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2,
            std::vector<Eigen::MatrixXd>& mat1_buf,
            std::vector<Eigen::MatrixXd>& mat2_buf,
            int batch_size) {

        if (mat1.rows() != mat2.rows()) {
            std::cerr << "Not possible to split, dimension problem!" << std::endl;
        }

        mat1_buf.clear();
        mat2_buf.clear();

        int number_of_matrices = mat1.rows() / batch_size;
        int current_row_index = 0;

        for (int i = 0; i < number_of_matrices; i++) {
            mat1_buf.push_back(mat1.block(current_row_index, 0, batch_size, mat1.cols()));
            mat2_buf.push_back(mat2.block(current_row_index, 0, batch_size, mat2.cols()));
            current_row_index += batch_size;
        }
    }

    Eigen::MatrixXd binomial(int rows, int cols, double ratio) {

        std::default_random_engine generator;
        std::binomial_distribution<int> distribution(1, ratio);

        Eigen::MatrixXd result(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result(r, c) = distribution(generator);
            }
        }
        return result;
    }
}

#endif	/* RANDOME_H */

