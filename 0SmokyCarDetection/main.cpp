#include "stdafx.h"
#include "dnnClassification.h"
#include "method.h"
#include <queue>
//#include <filesystem>
//#include "dirent.h"

#define SaveVideo
constexpr bool DISPLAY = 0;//�Ƿ���ʾ�Ŀ���
constexpr bool SAMPLE = 0;//�Ƿ��ȡ�����Ŀ���

int main(int argc, char** argv)
{
	Method method;
	std::pair<Backend,Target> p1 = (getAvailableBackends())[0];
	std::cout << p1.first<< "   "<< p1.second<< std::endl;
	std::string SrcPath = "E:\\video-1004";
	std::string ResultPath = "D:\\Result";
	//std::string inputVideo = findFile("E:/2018-10-02_170017.mp4");

	std::string objDetecMethod = "mobileNet_SSD"; //"Yolo_tiny_416_N23";
	std::string imgClassifyMethod = "CatsDogs190524";// "original version";

	int nCols = 1024; //1280;// ������Ƶ�ߴ�
	int	nRows = 768; //720;// 

	dnnObject carDetection(objDetecMethod);
	std::cout << objDetecMethod << "����Ŀ�����ʼ��  OK." << std::endl;
	dnnClassification smokyClassfy(imgClassifyMethod);
	std::cout << imgClassifyMethod << "���̷����ʼ��  OK." << std::endl;
	
	// Create a window
	static const std::string kWinName = "Smoky Car Detection";
	if (DISPLAY) namedWindow(kWinName, WINDOW_NORMAL);

	std::vector<std::string> fileNames;
	std::string fullVideoName;//�ļ�������׺
	std::string srcVideoFullPath;//ԭ��Ƶȫ·��
	std::string srcVideoName;//�ļ���������


	////��ȡ��Դ·���µ������ļ�  
	getFiles(SrcPath, fileNames);
	std::cout << "���������Ƶ����:" << fileNames.size() << std::endl;
	for (int i = 0; i < fileNames.size(); i++)
	{
		getTime();
		std::vector<std::string> result = split(fileNames[i].c_str(), "\\");
		fullVideoName = result[result.size() - 1];
		srcVideoFullPath = fileNames[i].c_str();
		std::cout << "��Ƶ:" << fullVideoName << std::endl;
		std::cout << "��Ƶ����·��:" << srcVideoFullPath << std::endl;
		result.clear();
		result = split(fullVideoName, ".");
		srcVideoName = result[0];
		std::cout << "���е���Ƶ��:" << srcVideoName << std::endl;

		std::string FolderPath = ResultPath + "\\" + srcVideoName;
		CreateDir(FolderPath.c_str());

		// Open a video file or an image file or a camera stream.
		VideoCapture cap;
		if (srcVideoFullPath != "") {
			cap.open(srcVideoFullPath);
		}
		else
			cap.open(0);

#ifdef SaveVideo
		VideoWriter writer;//��Ϊ��Ƶ
		VideoWriter smokyCarWriter;
		std::queue<Mat> frameSave;//���̳�����Ƶ
		int videoNum = 1; // �����Ƶ��˳��
		std::string outVideoName = ResultPath + "\\" + srcVideoName + "\\" + objDetecMethod+ "-"+ imgClassifyMethod + ".avi";
		writer.open(outVideoName, VideoWriter::fourcc('D', 'I', 'V', 'X'), 25.0, Size(nCols, nRows)); //VideoWriter::fourcc('D', 'I', 'V', 'X')
#endif
																								  // Process frames.
		bool isCatch = false;  // ����Ƿ�ʶ������̳�
		int frameNumber = 0;  // ͳ��֡��
		Mat frame;
		while (1) 
		{
			if (DISPLAY) waitKey(1);//��ʾ��
			cap >> frame;
			if (frame.empty())
			{
				break;
			}
			frameNumber++; // ֡������

			carDetection.run(frame);
			carDetection.postprocess(frame);

			std::vector<Rect> rectTrucksRear;//����β������
			carDetection.getTrucksRear(frame, rectTrucksRear);
			int res = 0;
			float classConfid = 0.0;

			for (int i = 0; i < rectTrucksRear.size(); i++)
			{
				Mat tmp = frame(rectTrucksRear[i]);
				if(DISPLAY) imshow("tmp", tmp);
				smokyClassfy.classify(tmp, res, classConfid);
				if (res == 1)
				{
					method.judgeSomkeCars(frame, rectTrucksRear[i], frameNumber, isCatch);
					rectangle(frame, rectTrucksRear[i], Scalar(0, 0, 255), 3);//��frame�ϻ����
				}
				else {
					rectangle(frame, rectTrucksRear[i], Scalar(0, 255, 255), 3);//��frame��Ҳ���ƿ�
				}
			}
			carDetection.afterprocess(frame);
			if(DISPLAY) carDetection.efficiencyInformation(frame);

#ifdef SaveVideo
			cv::resize(frame, frame, Size(nCols, nRows));

			if (frameSave.size() > 40)  frameSave.pop();//���̳�����Ƶ
			frameSave.push(frame.clone());
			writer << frame;
			// ������Ƶ
			if (isCatch == true)
			{
				static int videoFrames = 0;
				if (!smokyCarWriter.isOpened())
				{
					std::string smokyCarVideoName = ResultPath + "\\" + srcVideoName + "\\" + std::to_string(videoNum) + ".avi";
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
				}
			}
#endif 
			if(DISPLAY) imshow(kWinName, frame);
		}
#ifdef SaveVideo
		writer.release();
		std::cout << "Ŀǰ��⵽�ĺ��̳�����" << method.getBlackCarNum() << std::endl;
#endif 
	}
	return 0;
}
