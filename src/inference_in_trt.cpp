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
		
		initLibNvInferPlugins(&gLogger, "");
		TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
		assert(runtime != nullptr);
		
		mEngine = TRTUniquePtr<nvinfer1::ICudaEngine>(
			runtime->deserializeCudaEngine(trtModelStreamFromFile.data(), size_engine));
		assert(mEngine != nullptr);

		int max_batch_size = mEngine->getMaxBatchSize();
		cout << "max_batch_size: " << max_batch_size << endl;

		mContext = TRTUniquePtr<nvinfer1::IExecutionContext>(
			mEngine->createExecutionContext());
}

void TrtObjectDetector::extractContentsToBuffer(
	std::ifstream& file, std::vector <char>& buf) {
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

bool TrtObjectDetector::infer() {
	// get sizes of input and output and allocate memory required for input data and for output data
  std::vector<nvinfer1::Dims> input_dims; // we expect only one input
  std::vector<nvinfer1::Dims> output_dims; // and one output
  std::vector<void*> buffers(mEngine->getNbBindings()); // buffers for input and output data
  // for (size_t i = 0; i < mEngine->getNbBindings(); ++i) {
  //   auto binding_size = getSizeByDim(mEngine->getBindingDimensions(i)) * batch_size * sizeof(float);
  //   cudaMalloc(&buffers[i], binding_size);
  //   if (mEngine->bindingIsInput(i)) {
  //     input_dims.emplace_back(mEngine->getBindingDimensions(i));
  //   }
  //   else {
  //     output_dims.emplace_back(mEngine->getBindingDimensions(i));
  //   }
  // }
  if (input_dims.empty() || output_dims.empty()) {
	  std::cerr << "Expect at least one input and one output for network\n";
	  return false;
  }

  return true;
}	//end infer


TrtObjectDetector::~TrtObjectDetector() {
}



}	//namespace autocrane
