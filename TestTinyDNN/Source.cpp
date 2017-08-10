#include "Models.h"

using namespace std;


using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

void test_dnn_get_layer_types(void) {
	cout << "Function: " << __FUNCTION__ << endl;

	network<sequential> nn;
	nn << convolutional_layer(32, 32, 5, 3, 6) << activation::tanh()
		<< max_pooling_layer(28, 28, 6, 2) << activation::tanh()
		<< fully_connected_layer(14 * 14 * 6, 10) << activation::tanh();
	for (int i = 0; i < nn.depth(); i++) {
		cout << "layer:" << i << endl;
		cout << "layer types:" << nn[i]->layer_type() << endl;
		cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ")" << endl;
		cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() << ")" << endl;
	}
}

void dnn_iris(size_t iter) {
	cout << "Function: " << __FUNCTION__ << endl;
	string file_benchmark = ".\\data\\benchmark_dnn_iris.csv";

	// Load an iris dataset.
	vector<vec_t> X;
	vector<vec_t> y;

	if (load_iris_vec_t(X, y)) {
		ostringstream oss;

		int seed = 0;
		double train_ratio = 0.3;
		vector<vec_t> X_train, y_train, X_test, y_test;
		split_train_test(X, y, X_train, y_train, X_test, y_test, seed, train_ratio);

		timer t;

		oss << "BatchSize,Epoch,Time (sec.),Loss(Train),Accuracy(Train),Loss(Test),Accuracy(Test)" << endl;
		write_csv(file_benchmark, oss.str(), ios::trunc);
		cout << oss.str();
		oss.str("");
		oss.clear(stringstream::goodbit);

		// Define a model.
		network<sequential> model = create_model_1(X, y);

		// Train the model.
		adagrad opt;
		size_t batch_size = 10;
		size_t epoch = 10 * iter;
		int epoch_now = 0;
		
		t.start();
		model.fit<cross_entropy_multiclass>(opt, X_train, y_train, batch_size, epoch, 
			nop,
			[&]() {
			epoch_now++;
			if (epoch_now % 10 == 0) {
				oss << batch_size << ","
					<< epoch_now << ","
					<< fixed << setprecision(2) << t.elapsed() << ",";
				double loss_train = model.get_loss<cross_entropy_multiclass>(X_train, y_train);
				double accuracy_train = get_accuracy(model, X_train, y_train);
				double loss_test = model.get_loss<cross_entropy_multiclass>(X_test, y_test);
				double accuracy_test = get_accuracy(model, X_test, y_test);
				oss << fixed << setprecision(4) << loss_train << ","
					<< fixed << setprecision(4) << accuracy_train << ","
					<< fixed << setprecision(4) << loss_test << ","
					<< fixed << setprecision(4) << accuracy_test << endl;
				cout << oss.str();
				write_csv(file_benchmark, oss.str(), ios::app);
				oss.str("");
				oss.clear(stringstream::goodbit);
				t.restart();
			}
		});

		// Save the model.
		string modelname = ".\\data\\mymodel";
		model.save(modelname);

		t.stop();

	} else {
		cout << "Failure in loading data." << endl;
	}
}

void dnn_iris_load(void) {
	cout << "Function: " << __FUNCTION__ << endl;

	vector<vec_t> X;
	vector<vec_t> y;

	if (load_iris_vec_t(X, y)) {
		network<sequential> model;
		string modelname = ".\\data\\mymodel";
		model.load(modelname);

		int seed = 0;
		double train_ratio = 0.3;
		vector<vec_t> X_train, y_train, X_test, y_test;
		split_train_test(X, y, X_train, y_train, X_test, y_test, seed, train_ratio);

		ostringstream oss;
		oss << "Loss(Train),Accuracy(Train),Loss(Test),Accuracy(Test)" << endl;
		cout << oss.str();
		oss.str("");
		oss.clear(stringstream::goodbit);

		double loss_train = model.get_loss<cross_entropy_multiclass>(X_train, y_train);
		double accuracy_train = get_accuracy(model, X_train, y_train);
		double loss_test = model.get_loss<cross_entropy_multiclass>(X_test, y_test);
		double accuracy_test = get_accuracy(model, X_test, y_test);
		oss << fixed << setprecision(4) << loss_train << ","
			<< fixed << setprecision(4) << accuracy_train << ","
			<< fixed << setprecision(4) << loss_test << ","
			<< fixed << setprecision(4) << accuracy_test << endl;
		cout << oss.str();
		oss.str("");
		oss.clear(stringstream::goodbit);

	}
	else {
		cout << "Failure in loading data." << endl;
	}
}

void dnn_iris_2(size_t iter) {
	cout << "Function: " << __FUNCTION__ << endl;

	// Load an iris dataset.
	vector<vec_t> X;
	vector<vec_t> y;

	if (load_iris_vec_t(X, y)) {
		int seed = 0;
		double train_ratio = 0.3;
		vector<vec_t> X_train, y_train, X_test, y_test;
		split_train_test(X, y, X_train, y_train, X_test, y_test, seed, train_ratio);

		timer t;

		cout << "BatchSize, Epoch, Time, Loss(Train), Accuracy(Train), Loss(Test), Accuracy(Test)" << endl;
		for (size_t i = 1; i <= iter; i++) {
			// Define a model.
			network<sequential> model = create_model_2(X, y);

			// Train the model.
			adagrad opt;
			size_t batch_size = 10;
			size_t epoch = 10 * i;
			t.start();
			model.train<cross_entropy_multiclass>(opt, X_train, y_train, batch_size, epoch);
			t.stop();

			// Report.
			double loss_train = model.get_loss<cross_entropy_multiclass>(X_train, y_train);
			double accuracy_train = get_accuracy(model, X_train, y_train);
			double loss_test = model.get_loss<cross_entropy_multiclass>(X_test, y_test);
			double accuracy_test = get_accuracy(model, X_test, y_test);
			cout << batch_size << ", "
				<< epoch << ", "
				<< fixed << setprecision(2) << t.elapsed() << " sec., "
				<< fixed << setprecision(4) << loss_train << ", "
				<< fixed << setprecision(4) << accuracy_train << ", "
				<< fixed << setprecision(4) << loss_test << ", "
				<< fixed << setprecision(4) << accuracy_test << endl;
		}
	} else {
		cout << "Failure in loading data." << endl;
	}
}

void dnn_fx(size_t iter) {
	cout << "Function: " << __FUNCTION__ << endl;
	string file_benchmark = ".\\data\\benchmark_dnn_fx.csv";

	string fpath = ".\\data\\USDJPY-cd1_20170806_k030.csv";
	cout << "Input file: " << fpath << endl;

	// Load an iris dataset.
	vector<vec_t> X;
	vector<vec_t> y;

	if (load_data_vec_t(fpath, X, y, 1, 2, 3)) {
		ostringstream oss;

		int seed = 0;
		double train_ratio = 0.3;
		vector<vec_t> X_train, y_train, X_test, y_test;
		split_train_test(X, y, X_train, y_train, X_test, y_test, seed, train_ratio);

		timer t;

		oss << "BatchSize,Epoch,Time (sec.),Loss(Train),Accuracy(Train),Loss(Test),Accuracy(Test)" << endl;
		write_csv(file_benchmark, oss.str(), ios::trunc);
		cout << oss.str();
		oss.str("");
		oss.clear(stringstream::goodbit);

		// Define a model.
		network<sequential> model = create_model_1(X, y);

		// Train the model.
		adagrad opt;
		size_t batch_size = 100;
		size_t epoch = 10 * iter;
		int epoch_now = 0;

		t.start();
		model.fit<cross_entropy_multiclass>(opt, X_train, y_train, batch_size, epoch,
			nop,
			[&]() {
			epoch_now++;
			if (epoch_now % 10 == 0) {
				oss << batch_size << ","
					<< epoch_now << ","
					<< fixed << setprecision(2) << t.elapsed() << ",";
				double loss_train = model.get_loss<cross_entropy_multiclass>(X_train, y_train);
				double accuracy_train = get_accuracy(model, X_train, y_train);
				double loss_test = model.get_loss<cross_entropy_multiclass>(X_test, y_test);
				double accuracy_test = get_accuracy(model, X_test, y_test);
				oss << fixed << setprecision(4) << loss_train << ","
					<< fixed << setprecision(4) << accuracy_train << ","
					<< fixed << setprecision(4) << loss_test << ","
					<< fixed << setprecision(4) << accuracy_test << endl;
				cout << oss.str();
				write_csv(file_benchmark, oss.str(), ios::app);
				oss.str("");
				oss.clear(stringstream::goodbit);
				t.restart();
			}
		});
		t.stop();

	}
	else {
		cout << "Failure in loading data." << endl;
	}
}

void test_load_csv(string& fpath) {
	cout << "Function: " << __FUNCTION__ << endl;

	vector<vector<string>> dataset;
	bool success = load_csv(fpath, dataset);
	if (success) {
		for each (vector<string> var in dataset) {
			string str = "";
			for (int i = 0; i < (int)var.size(); i++) {
				str += var[i] + " ";
			}
			cout << str << endl;
		}
	} else {
		cout << "Failure in loading " << fpath << endl;
	}
}

void test_load_iris(void) {
	vector<vec_t> X;
	vector<vec_t> y;
	bool success = load_iris_vec_t(X, y);
	if (success) {
		for each (vec_t var in X) {
			string str = "";
			for (int i = 0; i < (int)var.size(); i++) {
				str += to_string(var[i]) + ",";
			}
			cout << str << endl;
		}
		for each (vec_t var in y) {
			string str = "";
			for (int i = 0; i < (int)var.size(); i++) {
				str += to_string(var[i]) + ",";
			}
			cout << str << endl;
		}
	} else {
		cout << "Failure in " << __FUNCTION__ << endl;
	}
}

void test_split_train_test(void) {
	cout << "seed?" << endl;
	string str;
	getline(cin, str);
	int seed = stoi(str);
	cout << "ratio?" << endl;
	getline(cin, str);
	double train_ratio = stod(str);

	vector<vec_t> X;
	vector<vec_t> y;
	if (load_iris_vec_t(X, y)) {
		vector<vec_t> X_train, y_train, X_test, y_test;
		split_train_test(X, y, X_train, y_train, X_test, y_test, seed, train_ratio);
		cout << "# of train data: " << X_train.size() << ", " << y_train.size() << endl;
		cout << "# of train data: " << X_test.size() << ", " << y_test.size() << endl;
	} else {
		cout << "Failure in loading data." << endl;
	}
}

void test_load_data_vec_t(void) {
	cout << "Function: " << __FUNCTION__ << endl;
	string fpath = ".\\data\\USDJPY-cd1_20170806_k001.csv";
	cout << "Input file: " << fpath << endl;
	vector<vec_t> X;
	vector<vec_t> y;
	bool success = load_data_vec_t(fpath, X, y, 1, 2, 3);
	if (success) {
		cout << X.size() << "," << y.size() << endl;
		/*
		for each (vec_t var in X) {
			string str = "";
			for (int i = 0; i < (int)var.size(); i++) {
				str += to_string(var[i]) + ",";
			}
			cout << str << endl;
		}
		for each (vec_t var in y) {
			string str = "";
			for (int i = 0; i < (int)var.size(); i++) {
				str += to_string(var[i]) + ",";
			}
			cout << str << endl;
		}
		*/
	}
	else {
		cout << "Failure in " << __FUNCTION__ << endl;
	}
}

bool load_csv(string& fpath, vector<vector<string>> &dst, int skiprows) {
	ifstream ifs(fpath);
	if (!ifs) {
		cout << "No such file:" << fpath << endl;
		return false;
	}
	int rows = 1;
	string str;
	while (getline(ifs, str)) {
		if ((rows++) <= skiprows) {
			continue;
		}
		dst.push_back(split(str, ','));
	}
	ifs.close();
	return true;
}

bool write_csv(string& fpath, const string str, int mode) {
	ofstream ofs;
	ofs.open(fpath, mode);
	if (!ofs) {
		cout << "No such file:" << fpath << endl;
		return false;
	}
	ofs << str;
	ofs.close();
	return true;
}

vector<string> split(string& str, char delimiter) {
	istringstream stream(str);
	string field;
	vector<string> result;
	while (getline(stream, field, delimiter)) {
		result.push_back(field);
	}
	return result;
}

bool load_iris_vec_t(vector<vec_t> &X, vector<vec_t> &y){
	// Load a dataset.
	string fpath = ".\\data\\iris.csv";
	vector<vector<string>> out_csv;
	if (!load_csv(fpath, out_csv)) {
		return false;
	}

	// Separate characteristics from labels.
	int n_char = 4;
	vector<string> labels;
	for each (vector<string> var in out_csv) {
		vec_t chars;
		for (int i = 0; i < n_char; i++) {
			try {
				chars.push_back(stof(var[i]));
			} catch (const std::invalid_argument &e) {
				cout << e.what() << endl;
				chars.push_back(0.0);
			}
		}
		X.push_back(chars);
		labels.push_back(var[n_char]);
	}

	// Convert labels into numeric labels.
	y = labeling_vec_t(labels);

	return true;
}

bool load_data_vec_t(string& fpath, vector<vec_t> &X, vector<vec_t> &y, int skiprows, int skipcols, int label_size) {
	// Load a dataset.
	vector<vector<string>> out_csv;
	if (!load_csv(fpath, out_csv, skiprows)) {
		return false;
	}

	// Separate characteristics from labels.
	int label_first = (int)out_csv[0].size() - label_size;
	vector<string> labels;
	for each (vector<string> var in out_csv) {
		// Characteristics.
		vec_t chars;
		for (int i = skipcols; i < label_first; i++) {
			try {
				chars.push_back(stof(var[i]));
			}
			catch (const std::invalid_argument &e) {
				cout << e.what() << endl;
				chars.push_back(0.0);
			}
		}
		X.push_back(chars);
		
		// Labels.
		vec_t label;
		for (int i = label_first; i < (int)out_csv[0].size(); i++) {
			try {
				label.push_back(stof(var[i]));
			}
			catch (const std::invalid_argument &e) {
				cout << e.what() << endl;
				label.push_back(0.0);
			}
		}
		y.push_back(label);
	}

	return true;
}

vector<vec_t> labeling_vec_t(vector<string> &y) {
	// Extract unique labels.
	vector<string> buff = vector<string>(y);
	sort(buff.begin(), buff.end());
	buff.erase(unique(buff.begin(), buff.end()), buff.end());

	// Convert.
	vector<vec_t> result;
	vector<string>::iterator it;
	for each (string str in y) {
		vec_t label = vec_t((int)buff.size(), 0);
		it = find(buff.begin(), buff.end(), str);
		int pos = (int)distance(buff.begin(), it);
		label[pos] = 1;
		result.push_back(label);
	}
	return result;
}

void split_train_test(const vector<vec_t> X, const vector<vec_t> y,
	vector<vec_t> &X_train, vector<vec_t> &y_train,
	vector<vec_t> &X_test, vector<vec_t> &y_test,
	int seed, double test_ratio) {
	
	mt19937 engine(seed);
	uniform_real_distribution<> dist(0.0, 1.0);
	vector<vec_t> x1, x2, y1, y2;
	for (int i = 0; i < X.size(); i++) {
		if (dist(engine) > test_ratio) {
			x1.push_back(X[i]);
			y1.push_back(y[i]);
		} else {
			x2.push_back(X[i]);
			y2.push_back(y[i]);
		}
	}
	X_train = x1; y_train = y1; X_test = x2; y_test = y2;
}


map<string, double> get_loss_accuracy(network<sequential> model,
	const vector<vec_t> X, const vector<vec_t> y) {
	map<string, double> result;

	double loss = model.get_loss<cross_entropy_multiclass>(X, y);
	double accuracy = get_accuracy(model, X, y);
	result["loss"] = loss;
	result["accuracy"] = accuracy;
	return result;
}

double get_accuracy(network<sequential> model,
	const vector<vec_t> X, const vector<vec_t> y) {
	double accuracy = 0.0;
	for (int i = 0; i < X.size(); i++) {
		label_t label = model.predict_label(X[i]);
		vec_t pred = vec_t(y[0].size(), 0.0);
		pred[label] = 1.0;
		vec_t answer = y[i];
		float dev = 0.0f;
		for (int j = 0; j < pred.size(); j++) {
			dev += (float)abs(pred[j] - answer[j]);
		}
		if (dev == 0.0f) {
			accuracy += 1.0;
		}
	}
	accuracy /= (double)X.size();
	return accuracy;
}