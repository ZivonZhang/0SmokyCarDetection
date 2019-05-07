#ifndef METHOD_H
#define METHOD_H


//#include "stdafx.h"

//using namespace std;
using namespace cv;

class Method
{
public:
	Method(){};
	~Method(){};
	

	void judgeSomkeCars(Mat &src, Rect &rect, int num, bool &isCatch);

	int getBlackCarNum(){ return blackCarNum; };
	
public:
	int out_count = 0;  //输出的车尾部图片的编号
	char file_name[100];

private:
	int count = 0; // 记录没发生变化的图像的帧数

	std::vector<Point2f> rectCenters;  //  记录每个检测到的黑烟车的框的中心点
	int sameNum = 0;  //  记录疑似同一辆黑烟车的个数
	int blackCarNum = 0;  //  识别出黑烟车个数
	int beforeFrameNum = 0;  // 上次检测到黑烟车的帧数
	char filename[100];  // 图片保存路径
};

#endif