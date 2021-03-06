#include "stdafx.h"
#include "dnnClassification.h"
#include "method.h"

//#include <filesystem>
//#include "dirent.h"

#define SaveVideo
constexpr bool DISPLAY = 0;//是否显示的开关
constexpr bool SAMPLE = 0;//是否获取样本的开关
constexpr bool SAVE_IMAGES = true;//是否获取待检的序列图

int main(int argc, char** argv)
{
	std::pair<Backend,Target> p1 = (getAvailableBackends())[0];
	std::cout << p1.first<< "   "<< p1.second<< std::endl; //这里可修改
	std::string SrcPath = "Z:\\Video\\北京测试视频";//"Y:\\Video\\北京测试视频\\temp";
	std::string ResultPath = "Z:\\result\\Groupmeeting1211";
	//std::string inputVideo = findFile("E:/2018-10-02_170017.mp4");

	Method method(ResultPath);
	 
	std::string objDetecMethod = "mobileNet_SSD"; //"Yolo_tiny_416_N23";
	std::string imgClassifyMethod = "mobilenet0731";//"CatsDogs190524";// 

	int nCols = 1024; //1280;// 保存视频尺寸
	int	nRows = 768; //720;// 

	dnnObject carDetection(objDetecMethod);
	std::cout << objDetecMethod << "车型目标检测初始化  OK." << std::endl;
	dnnClassification  smokyClassfy(imgClassifyMethod);
	std::cout << imgClassifyMethod << "黑烟分类初始化  OK." << std::endl;
	std::cout <<  "连续性阈值设定为  " << method.ContinuityThreshold << std::endl;
	
	// Create a window
	static const std::string kWinName = "Smoky Car Detection";
	if (DISPLAY) namedWindow(kWinName, WINDOW_NORMAL);

	std::vector<std::string> fileNames;
	std::string fullVideoName;//文件名带后缀
	std::string srcVideoFullPath;//原视频全路径
	std::string srcVideoName;//文件名不带后传


	////获取该源路径下的所有文件  
	getFiles(SrcPath, fileNames);
	std::cout << "将处理的视频个数:" << fileNames.size() << std::endl;

	CreateDir(ResultPath.c_str());//创建目标路径

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

		std::string FolderImgPath = FolderPath + "\\" + "smokyIMG";
		CreateDir(FolderImgPath.c_str());
		method.nowFolderPath = FolderImgPath;

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
		std::string outVideoName = ResultPath + "\\" + srcVideoName + "\\" + objDetecMethod+ "-"
			+ imgClassifyMethod +"-"+ std::to_string(method.ContinuityThreshold) + ".avi";
		writer.open(outVideoName, VideoWriter::fourcc('D', 'I', 'V', 'X'), 25.0, Size(nCols, nRows)); //VideoWriter::fourcc('D', 'I', 'V', 'X')
#endif
																								  // Process frames.
		bool isCatch = false;  // 标记是否识别出黑烟车
		int frameNumber = 0;  // 统计帧数
		Mat frame;
		while (1) 
		{
			if (DISPLAY) waitKey(1);//显示用
			cap >> frame;
			if (frame.empty())
			{
				break;
			}
			frameNumber++; // 帧数更新

			carDetection.run(frame);
			carDetection.postprocess(frame);

			std::vector<Rect> rectTrucksRear;//车辆尾部区域
			carDetection.getTrucksRear(frame, rectTrucksRear);
			int res = 0;
			float classConfid = 0.0;

			for (int i = 0; i < rectTrucksRear.size(); i++)
			{
				Mat tmp = frame(rectTrucksRear[i]);
				smokyClassfy.classify(tmp, res, classConfid);
				if (DISPLAY) smokyClassfy.efficiencyInformation(tmp);
				if (DISPLAY) imshow("tmp", tmp);
				if (res == 1)
				{
					method.judgeSomkeCars(frame, rectTrucksRear[i], frameNumber, isCatch);
					rectangle(frame, rectTrucksRear[i], Scalar(0, 0, 255), 3);//在frame上画红框
				}
				else {
					rectangle(frame, rectTrucksRear[i], Scalar(0, 255, 255), 3);//在frame上也画黄框
				}
			}
			carDetection.afterprocess(frame);
			if(DISPLAY) carDetection.efficiencyInformation(frame);

#ifdef SaveVideo
			cv::resize(frame, frame, Size(nCols, nRows));

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
				}
			}
#endif 
			if(DISPLAY) imshow(kWinName, frame);
		}
#ifdef SaveVideo
		writer.release();
		std::cout << "目前检测到的黑烟车数量为 " << method.getBlackCarNum() << std::endl;
#endif 
	}
	return 0;
}
