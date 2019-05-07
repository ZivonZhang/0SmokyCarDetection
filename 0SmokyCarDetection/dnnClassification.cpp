#include "stdafx.h"
#include "dnnClassification.h"


dnnClassification::dnnClassification(std::string method)
{
	if (method == "original version") {
		model = findFile("F:/Nets/net_iter_30.caffemodel");
		config = findFile("F:/Nets/deploy.prototxt");

		scale = 1.0;
		mean = { 0, 0, 0 };
		swapRB = false;
		inpWidth = 100;
		inpHeight = 100;
	}

	if (!classesFile.empty())	// Open file with classes names.
	{
		std::string file = classesFile;
		std::ifstream ifs(file.c_str());
		if (!ifs.is_open())
			CV_Error(Error::StsError, "File " + file + " not found");
		std::string line;
		while (std::getline(ifs, line))
		{
			classes.push_back(line);
		}
	}

	//! [Read and initialize network]
	net = readNet(model, config, framework);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
	//! [Read and initialize network]
	CV_Assert(!model.empty());
}

void dnnClassification::classify(Mat &region ,int &classNum , float &classConfid)
{
	Mat blob;
	//! [Create a 4D blob from a frame]
	blobFromImage(region, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
	//! [Create a 4D blob from a frame]

	//! [Set input blob]
	net.setInput(blob);
	//! [Set input blob]
	//! [Make forward pass]
	Mat prob = net.forward();
	//! [Make forward pass]

	//! [Get a class with a highest score]
	Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	//! [Get a class with a highest score]
	classNum = classId;
	classConfid = confidence;
	//std::string label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :classes[classId].c_str()),confidence);
}

void dnnClassification::efficiencyInformation(Mat & frame)
{
	// Put efficiency information.
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
}
