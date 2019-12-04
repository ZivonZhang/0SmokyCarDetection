#include "stdafx.h"
#include "dnnClassification.h"


dnnClassification::dnnClassification(std::string method)
{
	if (method == "original version") {
		model = findFile("F:/0SmokyCarDetection/model/net_iter_30.caffemodel");
		config = findFile("F:/0SmokyCarDetection/model/deploy.prototxt");

		scale = 1.0;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 100;
		inpHeight = 100;
	}
	if (method == "Tao") {
		model = findFile("F:/0SmokyCarDetection/model/tcnn3.caffemodel");
		config = findFile("F:/0SmokyCarDetection/model/train_val_tcnn3.prototxt");

		scale = 1.0;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 227;
		inpHeight = 227;
	}
	if (method == "CatsDogs190524") {
		model = findFile("F:/0SmokyCarDetection/model/catdognet_solver_iter_400.caffemodel");
		config = findFile("F:/0SmokyCarDetection/model/catdognet_deploy.prototxt");

		scale = 1.0;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 100;
		inpHeight = 100;
	}
	if (method == "Resnet190524") {
		model = findFile("F:/0SmokyCarDetection/model/ResNet-18-solver_iter_400.caffemodel");
		config = findFile("F:/0SmokyCarDetection/model/ResNet-18-deploy.prototxt");

		scale = 1.0;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 100;
		inpHeight = 100;
	}
	if (method == "mobilenet224") {
		model = findFile("F:/0SmokyCarDetection/model/mobile0627_loss_1838_17_optimized_graph.pb");
		config = "F:/0SmokyCarDetection/model/mobile0627_loss_1838_17_optimized_graph.pbtxt";

		scale = 0.007843;
		mean = { 127.5, 127.5, 127.5 };
		swapRB = true;
		inpWidth = 224;
		inpHeight = 224;
	}

	if (method == "mobilenetV2-224-0713") {
		model = findFile("F:/0SmokyCarDetection/model/mobilev2_1076_9535_loss_7_optimized_graph.pb");
		//config = findFile("F:/0SmokyCarDetection/model/mobileV1_avg_6_9570_optimized_graph.pbtxt");

		scale = 1.0/255;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 224;
		inpHeight = 224;
	}
	if (method == "mobilenet0713-best") {
		model = findFile("F:/0SmokyCarDetection/model/mobile_0713_best.pb");
		//config = findFile("F:/0SmokyCarDetection/model/mobileV1_avg_6_9570_optimized_graph.pbtxt");

		scale = 1.0 / 255;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 224;
		inpHeight = 224;
	}
	if (method == "mobilenet0715-best") {
		model = findFile("F:/0SmokyCarDetection/model/mobile_0715_best.pb");
		//config = findFile("F:/0SmokyCarDetection/model/mobileV1_avg_6_9570_optimized_graph.pbtxt");

		scale = 1.0 / 255;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 224;
		inpHeight = 224;
	}

	if (method == "mobilenet-before") {
		model = findFile("F:/0SmokyCarDetection/model/mobileV1_avg_6_9570_optimized_graph.pb");
		config = "F:/0SmokyCarDetection/model/mobileV1_avg_6_9570_optimized_graph.pbtxt";

		scale = 0.007843;
		mean = { 127.5, 127.5, 127.5 };
		swapRB = true;
		inpWidth = 224;
		inpHeight = 224;
	}

	if (method == "mobilenet0731") {
		model = findFile("F:/0SmokyCarDetection/model/20190731-132751.pb");
		//config = "F:/0SmokyCarDetection/model/mobileV1_avg_6_9570_optimized_graph.pbtxt";

		scale = 1.0 / 255;
		mean = { 0, 0, 0 };
		swapRB = true;
		inpWidth = 224;
		inpHeight = 224;
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
	if ("" == config) {
		net = readNetFromTensorflow(model);
	}
	else {
		net = readNet(model, config, framework);
	}
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
	if (1 != *prob.size) { //输出不为一维的情况
		minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
		int classId = classIdPoint.x;
		//! [Get a class with a highest score]
		classNum = classId;
		classConfid = confidence;
		//std::string label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :classes[classId].c_str()),confidence);
	}
	else {
		//std::array<float> temp = prob.reshape(1, 1);
		//float temp = prob(0, 0, CV_32FC1);
		float temp = prob.at<float>(0, 0);
		if (temp > ConfidThreshold) {
			classNum = 1;

		}
		else {
			classNum = 0;
		}
		classConfid = temp;
	}
	
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
