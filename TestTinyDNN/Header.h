#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>

#include "tiny_dnn\tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;

void test_dnn_get_layer_types(void);

void dnn_iris(size_t iter = 10);

void dnn_iris_2(size_t iter = 10);

void test_load_csv(string& fpath);

void test_load_iris(void);

void test_split_train_test(void);

void test_load_data_vec_t(void);

/***
	Load a dataset from a path of file.
	@param fpath   input file path
	@param dst     destination of the dataset
	@return        success (true) of failure (false) in loading
***/
bool load_csv(string& fpath, vector<vector<string>> &dst, int skiprows = 0);

/***
	Write a dataset to a file.
	@param fpath   output file path
	@param str     string to write
	@param mode    mode of opening the file
	@return        success (true) of failure (false) in writing
***/
bool write_csv(string& fpath, const string str, int mode = ios::trunc);

/***
	Split a string by a delimiter.
	@param str          string to split
	@param delimiter    delimiter
	@return             a vector instance with split strings
***/
vector<string> split(string& str, char delimiter);

/***
	Load a dataset of iris characteristics and labels.
	@param X       destination of characteristics
	@param y       destination of labels
	@return        success (true) of failure (false) in loading
***/
bool load_iris_vec_t(vector<vec_t> &X, vector<vec_t> &y);

/***
	Load a dataset of characteristics and labels.
	@param fpath       input file path
	@param X           destination of characteristics
	@param y           destination of labels
	@param skiprows    # of rows to skip from the 1st
	@param skipcols    # of columns to skip from the 1st
	pparam label_size  size of label
	@return            success (true) of failure (false) in loading
***/
bool load_data_vec_t(string& fpath, vector<vec_t> &X, vector<vec_t> &y, int skiprows = 0, int skipcols = 0, int label_size = 3);

/***
	Convert each label into vactor.
	@param y      labels
	@return       converted labels
***/
vector<vec_t> labeling_vec_t(vector<string> &y);

/***
	Split input datasets into train and test datasets.
	@param X            characteristics
	@param y            labels
	@param X_train      training characteristics
	@param y_train      training labels corresponding to X_train
	@param X_test       test characteristics
	@param y_test       test labels corresponding to X_test
	@param seed         seed of random number
	@param test_ratio   ratio of # of test datasets
***/
void split_train_test(const vector<vec_t> X, const vector<vec_t> y, 
	vector<vec_t> &X_train, vector<vec_t> &y_train, 
	vector<vec_t> &X_test, vector<vec_t> &y_test, 
	int seed = 0, double test_ratio = 0.3);

/***
	Return the loss and accuracy of a model to a dataset.
	@param model    a network model
	@param X        characteristics
	@param y        labels
	@return         map object <"loss": loss, "accuracy": accuracy>
***/
map<string, double> get_loss_accuracy(network<sequential> model,
	const vector<vec_t> X, const vector<vec_t> y);

/***
	Calculate the accuracy of a model to a dataset.
	@param model    a network model
	@param X        characteristics
	@param y        labels
	@return         accuracy
***/
double get_accuracy(network<sequential> model,
	const vector<vec_t> X, const vector<vec_t> y);