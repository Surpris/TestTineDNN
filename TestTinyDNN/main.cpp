#include "Header.h"

using namespace std;

int main(void) {

	try {
		timer t_all;
		cout << "start." << endl;
		t_all.start();
		//test_dnn_get_layer_types();

		//test_split_train_test();
		//string fpath = ".\\data\\iris.csv";
		//test_load_csv(fpath);

		//test_load_iris();
		
		dnn_iris(10);
		dnn_iris_load();

		//test_load_data_vec_t();
		//dnn_fx(10);
		t_all.stop();
		cout << "Elapsed time to process all:" << fixed << setprecision(2) << t_all.elapsed() << " sec." << endl;
	} catch (exception e) {
		cout << e.what() << endl;
	}

	cout << "Enter>";
	cin.ignore();
	return 0;
}