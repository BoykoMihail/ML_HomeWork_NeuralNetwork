/* 
 * File:   Evaluation.h
 * Author: boyko_mihail
 *
 * Created on 3 ноября 2019 г., 00:02
 */

#ifndef EVALUATION_H
#define	EVALUATION_H

#include "Confusion.h"

class Evaluation {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixEi;
public:
    double _avgAccuray;
    double _errRate;
    double _precisionMicro;
    double _recallMicro;
    double _fscoreMicro;
    double _precisionMacro;
    double _recallMacro;
    double _fscoreMacro;

    Evaluation() {
        _avgAccuray = 0.0;
        _errRate = 0.0;
        _precisionMicro = 0.0;
        _recallMicro = 0.0;
        _fscoreMicro = 0.0;
        _precisionMacro = 0.0;
        _recallMacro = 0.0;
        _fscoreMacro = 0.0;
    }

    Evaluation(
            double avgAccuray,
            double errRate,
            double precisionMicro,
            double recallMicro,
            double fscoreMicro,
            double precisionMacro,
            double recallMacro,
            double fscoreMacro
            ) :
    _avgAccuray(avgAccuray),
    _errRate(errRate),
    _precisionMicro(precisionMicro),
    _recallMicro(recallMicro),
    _fscoreMicro(fscoreMicro),
    _precisionMacro(precisionMacro),
    _recallMacro(recallMacro),
    _fscoreMacro(fscoreMacro) {
    }

    Evaluation(Confusion confusion) {
        _avgAccuray = 0.0;
        _errRate = 0.0;
        _precisionMicro = 0.0;
        _recallMicro = 0.0;
        _fscoreMicro = 0.0;
        _precisionMacro = 0.0;
        _recallMacro = 0.0;
        _fscoreMacro = 0.0;

        evaluation(confusion);
    }

    void evaluation(Confusion &confusion) {
        MatrixEi per(confusion._per);
        int numClasses = confusion._classes;

        //Average Accuracy
        double avgAccuracy = 0.0;

        //Error Rate 
        double errRate = 0.0;

        //Precision-Macro 
        double precisionMacro = 0.0;

        //Recall-Micro 
        double recallMacro = 0.0;

        double beta = 1;
        double fn = 0.0, fp = 0.0, tp = 0.0, tn = 0.0;

        double pNumerator = 0.0;
        double pDenominator = 0.0;

        double rNumerator = 0.0;
        double rDenominator = 0.0;

        for (int i = 0; i < numClasses; i++) {
            fn = per.coeff(i, 0);
            fp = per.coeff(i, 1);
            tp = per.coeff(i, 2);
            tn = per.coeff(i, 3);
            avgAccuracy = +avgAccuracy + ((tp + tn) / (tp + fn + fp + tn));
            errRate = +errRate + ((fp + fn) / (tp + fn + fp + tn));
            precisionMacro = precisionMacro + (tp / (tp + fp));
            recallMacro = recallMacro + (tp / (tp + fn));

            pNumerator = pNumerator + tp;
            pDenominator = pDenominator + (tp + fp);

            rNumerator = rNumerator + tp;
            rDenominator = rDenominator + (tp + fn);
        }
        avgAccuracy = avgAccuracy / numClasses;
        errRate = errRate / numClasses;
        precisionMacro = precisionMacro / numClasses;
        recallMacro = recallMacro / numClasses;

        //Fscore-Macro 
        double fscoreMacro =
                (((beta * beta) + 1) * precisionMacro * recallMacro) / ((beta * beta) * precisionMacro + recallMacro);

        //Precision-Micro 
        double precisionMicro = pNumerator / pDenominator;

        //Recall-Micro 
        double recallMicro = rNumerator / rDenominator;

        //Fscore-Micro 
        double fscoreMicro =
                (((beta * beta) + 1) * precisionMicro * recallMicro) / ((beta * beta) * precisionMicro + recallMicro);

        _avgAccuray = avgAccuracy;
        _errRate = errRate;
        _precisionMicro = precisionMicro;
        _recallMicro = recallMicro;
        _fscoreMicro = fscoreMicro;
        _precisionMacro = precisionMacro;
        _recallMacro = recallMacro;
        _fscoreMacro = fscoreMacro;
    }

    void print() {
        cout << "Accuracy Evaluation Results" << endl;
        cout << "=======================================" << endl;
        cout << "\tAverage System Accuracy(%): " << _avgAccuray * 100 << endl;
        cout << "\tSystem Error(%)           : " << _errRate * 100 << endl;
        cout << "\tPrecision (Micro)(%)      : " << _precisionMicro * 100 << endl;
        cout << "\tRecall (Micro)(%)         : " << _recallMicro * 100 << endl;
        cout << "\tFscore (Micro)(%)         : " << _fscoreMicro * 100 << endl;
        cout << "\tPrecision (Macro)(%)      : " << _precisionMacro * 100 << endl;
        cout << "\tRecall (Macro)(%)         : " << _recallMacro * 100 << endl;
        cout << "\tFscore (Macro)(%)         : " << _fscoreMacro * 100 << endl;
    }
};

#endif	/* EVALUATION_H */

