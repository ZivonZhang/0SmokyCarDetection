#include "stdafx.h"
#pragma once
#ifndef dnnObject_H
#define dnnObject_H


using namespace cv;
using namespace dnn;
class dnnObject
{
public:
	dnnObject(std::string method);
	~dnnObject() {};

	void run(Mat &frame);
	void postprocess(Mat &frame);
	void afterprocess(Mat &frame);
	void getTrucksRear(Mat & frame, std::vector<Rect>& out_rects);
	std::vector<int> getCar();
	Mat RectToMat(Rect rect, Mat frame);
	void efficiencyInformation(Mat &frame);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
	Rect getCarRear(Rect carbox);




private:
	std::vector<int> indices;
	float confThreshold = 0.35;
	float nmsThreshold = 0.4;
	float scale = 0.007843;
	Scalar mean = { 127.5, 127.5, 127.5 };
	bool swapRB = false;
	int inpWidth = 300;
	int inpHeight = 300;
	int backend = 0;
	int target = 0;
	int device = 0;
	std::string framework = "";
	std::string modelPath = "";
	std::string configPath = "";
	std::string classesFile = "";

	Net net;
	std::vector<String> outNames;
	std::vector<Mat> outs;

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;
	std::vector<std::string> classes;


};


#endif