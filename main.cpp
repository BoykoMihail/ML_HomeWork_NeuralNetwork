/* 
 * File:   main.cpp
 * Author: boyko_mihail
 *
 * Created on 28 октября 2019 г., 8:42
 */

#include <cstdlib>
#include <iostream>
#include "MnistUtil.h"
#include "Network.h"
#include "FullyConnected.h"
#include "DropOut.h"
#include "Identity.h"
#include "ReLU.h"
#include "SoftPlus.h"
#include "LeakyRelu.h"
#include "Softmax.h"
#include "LogSoftMax.h"
#include "MultiClassEntropy.h"
#include "AdaGrad.h"
#include "VerboseCallback.h"
#include "Confusion.h"
#include "Evaluation.h"

using namespace std;

int main(int argc, char** argv) {

    std::cout << "Mnist Dropout Activations Experiment Run..." << std::endl;

    int total_size = 60000;
    int test_size = 10000;

    MnistUtil m;

    Eigen::MatrixXd train_input = m.readMnistInput("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/ML_HomeWork_NeuralNetwork/Data/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXd train_output = m.readMnistOutput("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/ML_HomeWork_NeuralNetwork/Data/train-labels.idx1-ubyte", total_size);

    utilites::shuffleMatrixPair(train_input, train_output);

    Eigen::MatrixXd test_input = m.readMnistInput("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/ML_HomeWork_NeuralNetwork/Data/t10k-images.idx3-ubyte", test_size);
    Eigen::MatrixXd test_output = m.readMnistOutput("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/ML_HomeWork_NeuralNetwork/Data/t10k-labels.idx1-ubyte", test_size);


    Network net;

    Layer* layer1 = new FullyConnected<LeakyRelu>(784, 300);
    DropOut* layer2 = new DropOut(300, 300);
    layer2->set_dropout_ratio(0.9);
    Layer* layer3 = new FullyConnected<LeakyRelu>(300, 100);
    Layer* layer7 = new FullyConnected<Softmax>(100, 10);
    net.set_output(new MultiClassEntropy());
    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);
    net.add_layer(layer7);

    AdaGrad opt;
    opt.m_lrate = 0.001;

    VerboseCallback callback;
    net.set_callback(callback);

    net.init(0, 0.01, 123);

    cout << train_input.transpose().rows() << " " << train_input.transpose().cols() << endl;
    cout << train_output.transpose().rows() << " " << train_output.transpose().cols() << endl;
    cout << test_input.transpose().rows() << " " << test_input.transpose().cols() << endl;
    cout << test_output.transpose().rows() << " " << test_output.transpose().cols() << endl;

    //    net.check_gradient(train_input.transpose(), train_output.transpose(), 6, 123);

    net.fit(opt, train_input.transpose(), train_output.transpose(), 4000, 30, 123);

    auto pred = net.predict(test_input.transpose());

    cout << "Confusion" << endl;
    cout << test_output.transpose().rows() << endl;
    cout << pred.rows() << endl;

    cout << test_output.transpose().cols() << endl;
    cout << pred.cols() << endl;

    Confusion confusion = Confusion(test_output.transpose(), pred);
    confusion.print();

    Evaluation evaluation = Evaluation(confusion);
    evaluation.print();

    cout << " Mnist end \n";

    return 0;
}

