#include "stdafx.h"
#pragma once
#ifndef dnnClassification_H
#define dnnClassification_H


using namespace cv;
using namespace dnn;
class dnnClassification
{
public:
	dnnClassification(std::string method);
	
	~dnnClassification() {};
	void classify(Mat & region, int & classNum, float & classConfid);
	//void run(Mat &frame);
	//void postprocess(Mat &frame);
	//void afterprocess(Mat &frame);
	void efficiencyInformation(Mat &frame);
	double ConfidThreshold = 0.5;

private:

	float scale = 0.007843;
	Scalar mean = { 127.5, 127.5, 127.5 };
	bool swapRB = false;
	int inpWidth = 128;
	int inpHeight = 128;
	int backendId = 0;
	int targetId = 0;
	int deviceId = 0;
	std::string model;
	std::string config;
	std::string framework;
	std::string classesFile;

	Net net;
	std::vector<std::string> classes;
};


#endif