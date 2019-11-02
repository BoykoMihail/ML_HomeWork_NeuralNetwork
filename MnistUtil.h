/* 
 * File:   MnistUtil.h
 * Author: boyko_mihail
 *
 * Created on 28 октября 2019 г., 16:04
 */

#ifndef MNISTUTIL_H
#define	MNISTUTIL_H


#include <eigen3/Eigen/Core>
#include <vector>
#include <map>

class MnistUtil {
public:

    MnistUtil();

    Eigen::MatrixXd readMnistInput(const std::string& path,
            int number_of_items = 60000);

    Eigen::MatrixXd readMnistOutput(const std::string& path,
            int number_of_items = 60000);


private:
    int reverseInt(int i);

};

#endif	/* MNISTUTIL_H */
