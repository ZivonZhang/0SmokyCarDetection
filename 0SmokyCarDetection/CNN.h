#ifndef CNN_H
#define CNN_H

#define USE_OPENCV 1
#define CPU_ONLY 1

#include <caffe/common.hpp>
#include <caffe/layer.hpp>
#include <caffe/layer_factory.hpp>
#include <caffe/layers/input_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/dropout_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/softmax_layer.hpp> 
#include <caffe/layers/attention_crop_layer.hpp> 
#include <caffe/layers/power_layer.hpp> 
#include <caffe/layers/split_layer.hpp> 
#include <caffe/layers/tanh_layer.hpp>
#include <caffe/layers/sigmoid_layer.hpp> 
#include <caffe/layers/reshape_layer.hpp> 
#include <caffe/layers/concat_layer.hpp> 

#include <iostream>
#include <string>
#include <caffe/caffe.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <direct.h>
#include <ctime>

using namespace caffe;
using namespace cv;

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	//extern INSTANTIATE_CLASS(LRNLayer);
	//REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(AttentionCropLayer);
	//REGISTER_LAYER_CLASS(AttentionCrop);
	extern INSTANTIATE_CLASS(PowerLayer);
	//REGISTER_LAYER_CLASS(Power);
	extern INSTANTIATE_CLASS(SplitLayer);
	//REGISTER_LAYER_CLASS(Split);
	extern INSTANTIATE_CLASS(TanHLayer); 
	REGISTER_LAYER_CLASS(TanH);
	extern INSTANTIATE_CLASS(SigmoidLayer); 
	REGISTER_LAYER_CLASS(Sigmoid);
	extern INSTANTIATE_CLASS(ReshapeLayer); 
	extern INSTANTIATE_CLASS(ConcatLayer);
	//REGISTER_LAYER_CLASS(Sigmoid);
		
}

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file);

	int Classify(const cv::Mat& img);
	std::vector<float> Predict_new(const cv::Mat& img);

private:

	std::vector<int> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
};

int useCNN(int argc, Classifier &classifier);

#endif