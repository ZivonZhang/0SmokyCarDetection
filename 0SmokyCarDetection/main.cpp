#include "stdafx.h"
#include "dnnClassification.h"
#include "method.h"
#include <queue>
//#include "dirent.h"

#define SaveVideo

int main(int argc, char** argv)
{
	Method method;
	std::string ResultPath = "F:/Result";
	std::string inputVideo = findFile("E:/2018-10-02_170017.mp4");

	std::string objDetecMethod = "mobileNet_SSD"; //"Yolo_tiny_416_N23";
	std::string imgClassifyMethod = "original version";

	int nCols = 1024; //1280;// ������Ƶresize��������ƵҲ������ߴ�
	int	nRows = 768; //720;// 

	dnnObject carDetection(objDetecMethod);
	std::cout << objDetecMethod << "����Ŀ�����ʼ��  OK." << std::endl;
	dnnClassification smokyClassfy(imgClassifyMethod);
	std::cout << imgClassifyMethod << "���̷����ʼ��  OK." << std::endl;
	
	// Create a window
	static const std::string kWinName = "Smoky Car Detection";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Open a video file or an image file or a camera stream.
	VideoCapture cap;
	if (inputVideo != "") {
		cap.open(inputVideo);
	}
	else
		cap.open(0);

#ifdef SaveVideo
	VideoWriter writer;//��Ϊ��Ƶ
	VideoWriter smokyCarWriter;
	std::queue<Mat> frameSave;//���̳�����Ƶ
	int videoNum = 1; // �����Ƶ��˳��
	std::vector<std::string> temp = split(inputVideo, "/");
	std::vector<std::string> result = split(temp[1], ".");
	std::string outVideoName = ResultPath + "/" + result[0] + "_" + objDetecMethod + ".avi";
	writer.open(outVideoName, VideoWriter::fourcc('D', 'I', 'V', 'X'), 25.0, Size(nCols, nRows)); //VideoWriter::fourcc('D', 'I', 'V', 'X')

#endif
																								  // Process frames.
	bool isCatch = false;  // ����Ƿ�ʶ������̳�
	int frameNumber = 0;  // ͳ��֡��
	Mat frame;
	while (waitKey(1) < 0)
	{
		cap >> frame;
		if (frame.empty())
		{
			waitKey();
			break;
		}
		frameNumber++; // ֡������
		resize(frame, frame, Size(nCols, nRows));

		carDetection.run(frame);
		carDetection.postprocess(frame);

		std::vector<Rect> rectTrucksRear;//����β������
		carDetection.getTrucksRear(frame, rectTrucksRear);
		int res = 0;
		float classConfid = 0.0;

		for (int i = 0; i < rectTrucksRear.size(); i++)
		{
			Mat tmp = frame(rectTrucksRear[i]);
			imshow("tmp", tmp);
			smokyClassfy.classify(tmp, res, classConfid);
			if (res == 1)
			{
				method.judgeSomkeCars(frame, rectTrucksRear[i], frameNumber, isCatch);
				rectangle(frame, rectTrucksRear[i], Scalar(0, 0, 255), 2);//��frame��Ҳ����
			}
			else {
				rectangle(frame, rectTrucksRear[i], Scalar(0, 255, 255), 1);//��frame��Ҳ����
			}
		}
		carDetection.afterprocess(frame);
		carDetection.efficiencyInformation(frame);


		if (frameSave.size() > 40)  frameSave.pop();//���̳�����Ƶ
		frameSave.push(frame.clone());


#ifdef SaveVideo
		if (frameSave.size() > 40)  frameSave.pop();//���̳�����Ƶ
		frameSave.push(frame.clone());
		writer << frame;
		// ������Ƶ


		if (isCatch == true)
		{
			static int videoFrames = 0;
			if (!smokyCarWriter.isOpened())
			{
				std::string smokyCarVideoName = "F:/Result/smokycar/"  + std::to_string(videoNum) + ".avi";
				smokyCarWriter.open(smokyCarVideoName, VideoWriter::fourcc('D', 'I', 'V', 'X'), 25.0, Size(nCols, nRows));
			}
			smokyCarWriter << frameSave.front();
			videoFrames++;
			//if (videoFrames > 100 || frameNumber == nframes)
			if (videoFrames > 100)
			{
				smokyCarWriter.release();
				isCatch = false;
				videoFrames = 0;
				videoNum++;
				//sprintf(file, "resultVideo/blackCar%d.avi", videoNum);
				//writer.open(file, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(frameWidth, frameHeight));
			}
		}
#endif 

		imshow(kWinName, frame);
	}
#ifdef SaveVideo
	writer.release();
#endif 
	return 0;
}
