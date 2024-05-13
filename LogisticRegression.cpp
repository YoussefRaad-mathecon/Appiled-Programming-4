#include "armadillo.hpp"
#include <string>
#include <iostream>

// read data from a file into an Armadillo matrix
arma::mat readData(const std::string& filename) {
    arma::mat data;
    if (!data.load(filename)) {
        std::cerr << "Failed to load data from " << filename << std::endl;
    }
    return data;
}

// read labels from a file into an vector
arma::vec readLabels(const std::string& filename) {
    arma::vec labels;
    if (!labels.load(filename, arma::csv_ascii)) {
        std::cerr << "Failed to load labels from " << filename << std::endl;
    }
    return labels;
}

// perform logistic regression and return the optimized weights
arma::vec logisticRegression(const arma::mat& data, const arma::vec& labels, double alpha, double tolerance) {
    arma::vec weights = arma::zeros<arma::vec>(data.n_cols);
    bool converged = false;

    while (!converged) {
        arma::vec gradient = arma::zeros<arma::vec>(weights.size());
        double max_update = 0.0;

        for (size_t i = 0; i < data.n_rows; i++) {
            double wx = arma::dot(weights, data.row(i));
            double exp_term = std::exp(-labels(i) * wx);
            gradient += (-labels(i) * data.row(i).t()) / (1 + exp_term);
        }

        gradient /= data.n_rows;
        arma::vec update = alpha * gradient;
        weights -= update;
        max_update = arma::norm(update, "inf");

        if (max_update < tolerance) {
            converged = true;
        }
    }

    return weights;
}

// predict labels for test data 
arma::vec predictLabels(const arma::mat& data, const arma::vec& weights) {
    arma::vec predictions = data * weights;
    predictions.transform([](double val) { return val >= 0 ? 1.0 : -1.0; });
    return predictions;
}

// write predicted labels to a file
void writeLabels(const std::string& filename, const arma::vec& labels) {
    labels.save(filename, arma::csv_ascii);
}


int main() {
    const std::string trainDataFile = "dataX.dat";
    const std::string testDataFile = "dataXtest.dat";
    const std::string trainLabelFile = "dataY.dat";
    const std::string outputFile = "LogReg.dat";
    const double alpha = 0.01; // change if program isn't efficient enough
    const double tolerance = 1e-7;

    arma::mat trainData = readData(trainDataFile);
    arma::mat testData = readData(testDataFile);
    arma::vec trainLabels = readLabels(trainLabelFile);

    arma::vec weights = logisticRegression(trainData, trainLabels, alpha, tolerance);
    arma::vec predictedLabels = predictLabels(testData, weights);

    writeLabels(outputFile, predictedLabels);

    return 0;
}