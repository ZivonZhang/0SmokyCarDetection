#include "stdafx.h"
#include "dnnClassification.h"
#include "method.h"
#include <queue>
//#include <filesystem>
//#include "dirent.h"

#define SaveVideo
int main2()
{

	char * src_path = "E:\\video-1004";
	std::vector<std::string> fileNames;
	std::string fullVideoName;//文件名带后缀
	std::string srcVideoFullPath;//原视频全路径
	std::string srcVideoName;//文件名不带后传

	////获取该路径下的所有文件  
	getFiles(src_path, fileNames);

	for (int i = 0; i < fileNames.size(); i++)
	{
		//std::cout << fileNames[i].c_str() << std::endl;
		
		getTime();
		std::vector<std::string> result = split(fileNames[i].c_str(), "\\");
		fullVideoName = result[result.size()-1];
		srcVideoFullPath = fileNames[i].c_str();
		std::cout << "视频:" << fullVideoName << std::endl;
		std::cout << "视频完整路径:" << srcVideoFullPath << std::endl;
		result.clear();
		result = split(fullVideoName, ".");
		srcVideoName = result[0];
		std::cout << "剪切的视频名:" << srcVideoName << std::endl;
	}
	return 0;
}

int main(int argc, char** argv)
{
	Method method;
	std::string SrcPath = "E:\\video-1004";
	std::string ResultPath = "F:\\Result";
	//std::string inputVideo = findFile("E:/2018-10-02_170017.mp4");

	std::string objDetecMethod = "mobileNet_SSD"; //"Yolo_tiny_416_N23";
	std::string imgClassifyMethod = "original version";

	int nCols = 1024; //1280;// 输入视频resize，保存视频也是这个尺寸
	int	nRows = 768; //720;// 

	dnnObject carDetection(objDetecMethod);
	std::cout << objDetecMethod << "车型目标检测初始化  OK." << std::endl;
	dnnClassification smokyClassfy(imgClassifyMethod);
	std::cout << imgClassifyMethod << "黑烟分类初始化  OK." << std::endl;
	
	// Create a window
	static const std::string kWinName = "Smoky Car Detection";
	namedWindow(kWinName, WINDOW_NORMAL);


	std::vector<std::string> fileNames;
	std::string fullVideoName;//文件名带后缀
	std::string srcVideoFullPath;//原视频全路径
	std::string srcVideoName;//文件名不带后传


	////获取该源路径下的所有文件  
	getFiles(SrcPath, fileNames);
	for (int i = 0; i < fileNames.size(); i++)
	{
		getTime();
		std::vector<std::string> result = split(fileNames[i].c_str(), "\\");
		fullVideoName = result[result.size() - 1];
		srcVideoFullPath = fileNames[i].c_str();
		std::cout << "视频:" << fullVideoName << std::endl;
		std::cout << "视频完整路径:" << srcVideoFullPath << std::endl;
		result.clear();
		result = split(fullVideoName, ".");
		srcVideoName = result[0];
		std::cout << "剪切的视频名:" << srcVideoName << std::endl;

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
		VideoWriter writer;//存为视频
		VideoWriter smokyCarWriter;
		std::queue<Mat> frameSave;//黑烟车短视频
		int videoNum = 1; // 输出视频的顺序
		std::string outVideoName = ResultPath + "\\" + srcVideoName + "\\" + objDetecMethod+ ".avi";
		writer.open(outVideoName, VideoWriter::fourcc('D', 'I', 'V', 'X'), 25.0, Size(nCols, nRows)); //VideoWriter::fourcc('D', 'I', 'V', 'X')

#endif
																								  // Process frames.
		bool isCatch = false;  // 标记是否识别出黑烟车
		int frameNumber = 0;  // 统计帧数
		Mat frame;
		while (waitKey(1) < 0) 
		{
			cap >> frame;
			if (frame.empty())
			{
				break;
			}
			frameNumber++; // 帧数更新
			resize(frame, frame, Size(nCols, nRows));

			carDetection.run(frame);
			carDetection.postprocess(frame);

			std::vector<Rect> rectTrucksRear;//车辆尾部区域
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
					rectangle(frame, rectTrucksRear[i], Scalar(0, 0, 255), 2);//在frame上也画框
				}
				else {
					rectangle(frame, rectTrucksRear[i], Scalar(0, 255, 255), 1);//在frame上也画框
				}
			}
			carDetection.afterprocess(frame);
			carDetection.efficiencyInformation(frame);


			if (frameSave.size() > 40)  frameSave.pop();//黑烟车短视频
			frameSave.push(frame.clone());


#ifdef SaveVideo
			if (frameSave.size() > 40)  frameSave.pop();//黑烟车短视频
			frameSave.push(frame.clone());
			writer << frame;
			// 生成视频


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
	}
	return 0;
}
