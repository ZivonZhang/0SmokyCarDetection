#include "stdafx.h"
#include "method.h"

//using namespace cv;
//using namespace std;

Method::Method(std::string R) {
	ResultPath = R;
}

void Method::clear(std::queue<Mat>& q) {
	std::queue<Mat> empty;
	swap(empty, q);
}

/**************** judge smoke cars ********************/

void Method::judgeSomkeCars(Mat &src, Rect &rect, int frameNum, bool &isCatch)
{
	if (frameNum - beforeFrameNum > 20)  // 如果输入的两帧帧数差大于20，则认为是两辆车，清空储存信息
	{
		sameNum = 0;
		rectCenters.clear();
		std::vector<Point2f>(rectCenters).swap(rectCenters);
		clear(area_Smoke); //清空车尾区域
		out_count = 0;
	}

	Point2f newCenter;
	newCenter.x = rect.x + rect.width / 2;
	newCenter.y = rect.y + rect.height / 2;
	double distance = 100000;
	if (rectCenters.size() != 0)
	{
		distance = sqrt((rectCenters.back().x - newCenter.x)*(rectCenters.back().x - newCenter.x) + (rectCenters.back().y - newCenter.y)*(rectCenters.back().y - newCenter.y));
		if (distance < src.rows / 4) //距离差小于src.rows / 4，此处可改进为车辆长度相关，认为是同一辆车
		{ 
			sameNum++;
			area_Smoke.push(src(rect).clone());
		}
	}

	if (sameNum == ContinuityThreshold)  // 实际为检测出ContinuityThreshold帧才认为是成功检测
	{
		std::cout << "发现一辆黑烟车" << std::endl;
		isCatch = true;
		blackCarNum++;
		sameNum++;
	}

	while ( sameNum >= ContinuityThreshold && !area_Smoke.empty()) {
		out_count++;
		sprintf(filename, "/Smoky%03d-%02d.jpg", blackCarNum, out_count);
		imwrite(nowFolderPath + filename, area_Smoke.front());
		area_Smoke.pop();
	}

	if (distance > src.rows / 4 || sameNum > 70)
	{
		sameNum = 0;
		rectCenters.clear();
		std::vector<Point2f>(rectCenters).swap(rectCenters);
		clear(area_Smoke); //清空车尾区域
		out_count = 0;
	}
	rectCenters.push_back(newCenter);
	beforeFrameNum = frameNum;

	//waitKey(1);

}
