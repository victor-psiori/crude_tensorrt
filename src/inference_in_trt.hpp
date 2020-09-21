#ifndef AUTOCRANE_TRT_OBJDETECTOR
#define AUTOCRANE_TRT_OBJDETECTOR

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <NvInfer.h>
#include <NvInferPlugin.h>

//TensorRT common/ headers
// #include "argsParser.h"
// #include "buffers.h"
// #include "common.h"
// #include "logger.h"

namespace autocrane {

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

//destroy TensorRT objects if something goes wrong
struct TRTDestroy{
	template <typename T>
	void operator()(T* obj) const {
		if (obj) {
			obj->destroy();
		}
	}
};


template<typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

class TrtObjectDetector {
public:
	TrtObjectDetector(const std::string filename);
	virtual ~TrtObjectDetector();
	/**
	 * @brief function extracts content in .bin engine file to char vector
	 * @param file: ifstream object to which .bin engineFile is loaded in bin mode
	 * @param buf: char buffer to extracts contents of file
	*/
	void extractContentsToBuffer(std::ifstream& file, std::vector <char>& buf);

	bool infer();


private:
	std::string engineFile;
	std::size_t size_engine;
	// nvinfer1::ICudaEngine* mEngine;
	TRTUniquePtr<nvinfer1::ICudaEngine> mEngine;
	TRTUniquePtr<nvinfer1::IExecutionContext> mContext;
};

}; // namespace autocrane



#endif /* AUTOCRANE_TRT_OBJDETECTOR */