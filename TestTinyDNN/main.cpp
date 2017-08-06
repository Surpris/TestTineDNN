#include "Header.h"

using namespace std;

int main(void) {
	//test_dnn_get_layer_types();
	
	try {
		timer t_all;
		t_all.start();
		dnn_iris(10);
		t_all.stop();
		cout << "Elapsed time to process all:" << fixed << setprecision(2) << t_all.elapsed() << " sec." << endl;
	} catch (exception e) {
		cout << e.what() << endl;
	}
	

	//test_split_train_test();
	//string fpath = ".\\data\\iris.csv";
	//test_load_csv(fpath);

	//test_load_iris();
	cout << "Enter>";
	cin.ignore();
	return 0;
}