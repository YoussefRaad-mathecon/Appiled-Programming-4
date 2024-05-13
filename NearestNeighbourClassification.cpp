#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <vector>
#include "armadillo.hpp"

int main()
{
	std::vector <double> dataX;
	std::vector <double> dataY;
	std::vector <double> test;
	double storage;
	short storagei;
	int knn = 5;

	std::ifstream read_file("dataX.dat");
	assert(read_file.is_open());
	while (!read_file.eof())
	{
		read_file >> storage;
		dataX.push_back(storage);
	}
	read_file.close();

	read_file.open("dataY.dat");
	assert(read_file.is_open());
	while (!read_file.eof())
	{
		read_file >> storagei;
		dataY.push_back(storagei);
	}
	read_file.close();


	read_file.open("dataXtest.dat");
	assert(read_file.is_open());
	while (!read_file.eof())
	{
		read_file >> storage;
		test.push_back(storage);
	}
	read_file.close();

	int lenX;
	lenX = dataX.end() - dataX.begin() -1;
	int lenY;
	lenY = dataY.end() - dataY.begin() -1;
	int lenTest;
	lenTest = test.end() - test.begin() -1;
	int XY;
	XY = lenX / lenY;

	arma::mat X(lenY, XY, arma::fill::none);
	for (int i = 0; i < lenY; i++)
	{
		for (int j = 0; j < XY; j++)
		{
			X(i, j) = dataX[XY*i + j];
		}
	}

	int test_rows = lenTest / XY;

	arma::mat Xtest(test_rows, XY, arma::fill::none);
	for (int i = 0; i < test_rows; i++)
	{
		for (int j = 0; j < XY; j++)
		{
			Xtest(i, j) = test[XY*i + j];
		}
	}

	arma::Col<short> Y(lenY);
	for (int i = 0; i < lenY; i++)
	{
		Y(i) = dataY[i];
	}
	std::vector<double> distances(X.n_rows);
	std::vector<double> distances2(X.n_rows);
	double result[Xtest.n_rows];

	for (int g = 0; g < Xtest.n_rows; g++)
	{
		for (int j = 0; j < X.n_rows; j++)
		{
			double d = 0;
			for (int k = 0; k < X.n_cols; k++)
			{

				d += pow((Xtest(g, k) - X(j, k)), 2);
			}
			distances[j] = sqrt(d);
		}
		distances2 = distances;
		std::sort(distances2.begin(), distances2.end());
		int search;
		int storage[knn];
		for (int i = 0; i < knn; i++)
		{
			search = std::find(distances.begin(), distances.end(), distances2[i]) - distances.begin();
			storage[i] = Y(search);
		}
		int sum = 0;
		for (int i = 0; i < knn; i++)
		{
			sum += storage[i];
		}
		if (sum > 0)
		{
			result[g] = 1;
		}
		else
		{
			result[g] = -1;
		}

		std::ofstream write_file("NN.dat");
		for (int h = 0; h < Xtest.n_rows; h++)
		{
			write_file << result[h] << "\n";
		}
	}
	return 0;
}