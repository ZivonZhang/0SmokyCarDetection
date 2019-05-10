#include "stdafx.h"




std::string keys =
"{ config      | F:/SSD/Model/deploy.prototxt | Path to a text file of model contains network configuration. Itcould be a file with extensions.prototxt(Caffe), .pbtxt(TensorFlow), .cfg(Darknet), .xml(OpenVINO).. }"
"{ model       | F:/SSD/Model/carModel_SSD_300x300_iter_10000.caffemodel | Path to a binary file of model contains trained weights. }"

"{ help  h     | | Print help message. }"
"{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
"{ zoo         | | An optional path to file with preprocessing parameters }"
"{ device      |  0 | camera device number. }"
"{ input i     | D:/2017-03-17_120000.mp4 | Path to input image or video file. Skip this argument to capture frames from a camera. }"
"{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
"{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
"{ thr         | 0.35 | Confidence threshold. }"
"{ nms         | .4 | Non-maximum suppression threshold. }"
"{ backend     |  0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation }"
"{ target      | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU }";


using namespace cv;
using namespace dnn;




dnnObject::dnnObject(std::string method){//参数与网络初始化
	//default
	std::string classesFile = findFile("F:/SSD/Model/predefined_classes.txt");

	if (method == "Yolo_tiny_416_N23") {
		modelPath = findFile("F:/Yolo/yolov3-tiny-carTrunk_final.weights");
		configPath = findFile("F:/Yolo/yolov3-tiny-carTrunk.cfg");

		confThreshold = 0.5;
		nmsThreshold = 0.4;
		scale = 0.00392;
		mean = { 0,0,0 };
		swapRB = true;
		inpWidth = 416;
		inpHeight = 416;
	}

	if (method == "Yolo_416") {
		modelPath = findFile("F:/Yolo/yolov3-carTrunk_final.weights");
		configPath = findFile("F:/Yolo/yolov3-carTrunk.cfg");

		confThreshold = 0.5;
		nmsThreshold = 0.4;
		scale = 0.00392;
		mean = { 0,0,0 };
		swapRB = true;
		inpWidth = 416;
		inpHeight = 416;
	}

	if (method == "VGG_SSD") {
		modelPath = findFile("F:/SSD/Model/carModel_SSD_300x300_iter_10000.caffemodel");
		configPath = findFile("F:/SSD/Model/deploy.prototxt");
	}


	if (method == "mobileNet_SSD") {
		modelPath = findFile("F:/SSD/Model/mobilenet_iter_10000.caffemodel");
		configPath = findFile("F:/SSD/Model/MobileNetSSD_deploy.prototxt");

		confThreshold = 0.35;
		nmsThreshold = 0.4;

		scale = 0.007843;
		mean = { 127.5, 127.5, 127.5 };
		swapRB = false;
		inpWidth = 300;
		inpHeight = 300;
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
	// Load a model.
	net = readNet(modelPath, configPath, framework);
	net.setPreferableBackend(backend);
	net.setPreferableTarget(target);
	outNames = net.getUnconnectedOutLayersNames();

}

void dnnObject::run(Mat &frame)////预处理与前向计算
{
	// Create a 4D blob from a frame.
	Mat blob;
	Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
		inpHeight > 0 ? inpHeight : frame.rows);
	blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);

	// Run a model.
	net.setInput(blob);
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		resize(frame, frame, inpSize);
		Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		net.setInput(imInfo, "im_info");
	}

	net.forward(outs, outNames);
}

void dnnObject::postprocess(Mat &frame)//获得车辆所在区域与车型置信度
{
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	classIds.clear();
	confidences.clear();
	boxes.clear();


	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float* data = (float*)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > confThreshold)
			{
				int left = (int)data[i + 3];
				int top = (int)data[i + 4];
				int right = (int)data[i + 5];
				int bottom = (int)data[i + 6];
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "DetectionOutput")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float* data = (float*)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > confThreshold)
			{
				int left = (int)(data[i + 3] * frame.cols);
				int top = (int)(data[i + 4] * frame.rows);
				int right = (int)(data[i + 5] * frame.cols);
				int bottom = (int)(data[i + 6] * frame.rows);
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "Region")//YOLO
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}
	}
	else
		CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
}

void dnnObject::afterprocess(Mat & frame)//框出车辆
{
	//std::vector<int> indices;
	//NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,//框出车辆
			box.x + box.width, box.y + box.height, frame);
	}
}

void dnnObject::getTrucksRear(Mat &frame, std::vector<Rect> &out_rects) {
	indices.clear();
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		if (classes[classIds[idx]] == "trucks") {//如果为货车，检测尾部								
			Rect rect1, rect2;
			rect1 = box;  // 轮廓外接矩形
			//rect1.height = rect1.height * 1.2;  //rect1拉长包括黑烟检测区域
												//carRects.push_back(rect1);
			Point p1, p2, p3;
			p1.x = rect1.x;
			p1.y = rect1.y;
			p2.x = rect1.x + rect1.width;
			p2.y = rect1.y + rect1.height;   //p1为rect1左上角的点，p2为右下角

			p3.x = p1.x;
			//p3.y = rect1.y + rect1.height / 4 * 3;
			p3.y = rect1.y + rect1.height / 2;   //p3为rect2左上角


												 //if (rect1.width>=65)
			if ((rect1.width >= 50) && (rect1.height >= 50))//50   
				if (p1.x - 5 > 0 && p1.y - 5 > 0 && p2.x + 5 < frame.cols && p2.y + 5 < frame.rows)
				{
					if ((rect1.height > 100 && p1.y > frame.rows / 2) || (rect1.height > 100 && p1.y <= frame.rows / 2 && p1.y > frame.rows / 4) || (rect1.height > 70 && p1.y <= frame.rows / 4))//100 100 70
																																														  //if(1)
					{
						rect2.x = p3.x;
						rect2.y = rect1.y + 0.7 * rect1.height;

						rect2.width = rect1.width;
						rect2.height = rect1.height / 2;
						if (((rect2.width / rect2.height) < 2) && ((rect2.height / rect2.width) < 2))
						{
							//car_rects.push_back(rect3);//车型识别区域
							//out_images.push_back(src(rect2));
							if ((rect2.y + rect2.height) > frame.rows) continue;//防止框到外面
							out_rects.push_back(rect2);
							//rectangle(src, rect3, Scalar(0, 255, 255), 1);//画框
						}
					}
				}

		}
	}

}

std::vector<int> dnnObject::getCar()
{
	std::vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	return indices;
}


void dnnObject::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat & frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);//车辆框

	std::string label = format("%.2f", conf);
	if (!classes.empty())
	{
		int classSize = (int)classes.size();
		if (classId >= classSize) {//error
			printf("error:unknow class: %d\n", classId);
			classId = classSize - 1;
		}
		CV_Assert(classId < classSize);
		label = classes[classId] + ": " + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);//文本框

	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - labelSize.height),
		Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.6, Scalar());//文本
}

Rect dnnObject::getCarRear(Rect carbox)
{
	Rect carRear;
	Point p1, p2, p3;
	carbox.height *= 1.2;  //拉长
	p1.x = carbox.x;
	p1.y = carbox.y;
	p2.x = carbox.x + carbox.width;
	p2.y = carbox.y + carbox.height;   //p1为carbox左上角的点，p2为右下角

	p3.x = p1.x;
	//p3.y = rect1.y + rect1.height / 4 * 3;
	p3.y = carbox.y + carbox.height / 2;   //p3为carRear左上角

	carRear.x = p3.x;
	carRear.y = carbox.y + carbox.height / 4 * 3;
	carRear.width = carbox.width;
	carRear.height = carbox.height / 2;

	return carRear;
}

Mat dnnObject::RectToMat(Rect rect,Mat frame) {
		;
		return frame;
}
void dnnObject::efficiencyInformation(Mat &frame)
{
	// Put efficiency information.
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0));
}
