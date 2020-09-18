#include <NvInfer.h>
#include <iostream>

using namespace std;

class Logger : public nvinfer1::ILogger {
public:
	nvinfer1::ILogger& getTRTLogger() {
		return *this;
	}

	void log(Severity severity, const char* msg) override {
		//suppress info level messages
		if (severity != Severity::kINFO) {
			std::cout << msg << std::endl;
		}
	}

};

int main () {
	cout << "Hey hey" << endl;
}