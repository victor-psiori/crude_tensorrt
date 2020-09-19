#include "inference_in_trt.hpp"
#include <cassert>

using namespace std;

namespace autocrane {


TrtObjectDetector::TrtObjectDetector(const std::string filename) :
	engineFile(filename) {
		std::ifstream file(engineFile, std::ios::binary);
		vector<char> trtModelStreamFromFile;
		extractContentsToBuffer(file, trtModelStreamFromFile);
		Logger gLogger;
		// for plugin deserialization errors.
		// nvinfer1::initLibNvInferPlugins(&gLogger, "");
		// nvinfer1::IRuntime* runtime_ = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
		// std::unique_ptr<IRuntime, Destroy> runtime{createInferRuntime(gLogger)};


		TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
		assert(runtime != nullptr);

		// mEngine = runtime->nvinfer1::IRuntime::deserializeCudaEngine(
		// 	trtModelStreamFromFile.data(), size_engine);

		
		mEngine = TRTUniquePtr<nvinfer1::ICudaEngine>(
			runtime->deserializeCudaEngine(trtModelStreamFromFile.data(), size_engine));
}

void TrtObjectDetector::extractContentsToBuffer(std::ifstream& file,
																							   std::vector <char>& buf) {
	if (file.good()) {
		file.seekg(0, file.end);
		size_engine = file.tellg();
		file.seekg(0, file.beg);
		buf.resize(size_engine);
		std::cout << "size of engine file: " << buf.size() << std::endl;
		file.read(buf.data(), size_engine);
		file.close();
	}
}

TrtObjectDetector::~TrtObjectDetector() {
}



}	//namespace autocrane


int main () {
	cout << "Hey hey" << endl;
}
