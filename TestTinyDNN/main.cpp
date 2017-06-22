#include "Header.h"

using namespace std;

int main(void) {
	//test_dnn_get_layer_types();
	
	try {
		dnn_iris(10);
	} catch (exception e) {
		cout << e.what() << endl;
	}
	

	//test_split_train_data();
	//string fpath = ".\\data\\iris.csv";
	//test_load_csv(fpath);

	//test_load_iris();
	cout << "Enter>";
	cin.ignore();
	return 0;
}