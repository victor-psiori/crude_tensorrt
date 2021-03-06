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
		// using sample::gLogger;
		// initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
		// initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
		// nvinfer1::IRuntime* runtime = createInferRuntime(gLogger);
		// nvinfer1::IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
		nvinfer1::IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
		// TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
		assert(runtime != nullptr);
		
		// mEngine = TRTUniquePtr<nvinfer1::ICudaEngine>(
		// 	runtime->deserializeCudaEngine(trtModelStreamFromFile.data(), size_engine), 
		// 	samplesCommon::InferDeleter());
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
				runtime->deserializeCudaEngine(trtModelStreamFromFile.data(), size_engine),
			  samplesCommon::InferDeleter()
		  );
		assert(mEngine != nullptr);

		int max_batch_size = mEngine->getMaxBatchSize();
		cout << "max_batch_size: " << max_batch_size << endl;
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

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool TrtObjectDetector::processInput(const samplesCommon::BufferManager& buffers) {
	const int inputC = mInputDims.d[0];
	const int inputH = mInputDims.d[1];
	const int inputW = mInputDims.d[2];
	const int batchSize = 1;

	// Available images
	std::vector<std::string> imageList = {"grapple_1357.jpg"};
	mPPMs.resize(batchSize);
	assert(mPPMs.size() <= imageList.size());
	for (int i=0; i<batchSize; ++i) {
		// readPPMFile(locateFile(imageList[i], "../assets/"), mPPMs[i]);
		readPPMFile("../assets/grapple_1357.jpg", mPPMs[i]);
	}
	//fill data buffer
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("data"));
	float pixelMean[3]{104.0f, 117.0f, 123.0f}; // In BGR order
  // Host memory for input buffer
  int volImg = inputC * inputH * inputW;
  int volChl = inputH * inputW;
  for (int i = 0; i < batchSize; ++i) {
    for (int c = 0; c < inputC; ++c) {
      // The color image to input should be in BGR order
      for (unsigned j = 0; j < volChl; ++j) {
        hostDataBuffer[i * volImg + c * volChl + j] = float(mPPMs[i].buffer[j * inputC + 2 - c]);
      }
    }
  }

  return true;
}

bool TrtObjectDetector::infer() {
	// Create RAII buffer manager object
  samplesCommon::BufferManager buffers(mEngine, 1);

	auto mContext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (mContext == nullptr) {
		return false;
	}

	if (!processInput(buffers)) {
		return false;
	}

	//memcpy from host input buffers to device input buffers
	buffers.copyInputToDevice();

	bool status = mContext->execute(1, buffers.getDeviceBindings().data());
	if (!status) {
		return false;
	}

	//memcpy from device output buffers to host output buffers
	buffers.copyOutputToHost();
	return true;
}	//end infer

// calculate size of tensor
size_t TrtObjectDetector::getSizeByDim(const nvinfer1::Dims& dims) {
  size_t size = 1;
  for (size_t i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];
  }
  return size;
}


TrtObjectDetector::~TrtObjectDetector() {
}



}	//namespace autocrane
