#include <iostream>
#include "inference_in_trt.hpp"


using namespace autocrane;
using namespace std;

int main () {
	TrtObjectDetector* obj_detector = new TrtObjectDetector("../models/TRT_joel150.bin");
}