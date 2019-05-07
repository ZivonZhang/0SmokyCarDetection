#include "stdafx.h"
#include "method.h"

//using namespace cv;
//using namespace std;



/**************** judge smoke cars ********************/

void Method::judgeSomkeCars(Mat &src, Rect &rect, int frameNum, bool &isCatch)
{

	if (frameNum - beforeFrameNum > 20)  // 如果输入的两帧帧数差大于20，则认为是两辆车，清空储存信息
	{
		sameNum = 0;
		rectCenters.clear();
		std::vector<Point2f>(rectCenters).swap(rectCenters);
	}

	Point2f newCenter;
	newCenter.x = rect.x + rect.width / 2;
	newCenter.y = rect.y + rect.height / 2;
	double distance = 100000;
	if (rectCenters.size() != 0)
	{
		distance = sqrt((rectCenters.back().x - newCenter.x)*(rectCenters.back().x - newCenter.x) + (rectCenters.back().y - newCenter.y)*(rectCenters.back().y - newCenter.y));
		if (distance < 70)
		{
			sameNum++;
		}
	}

	if (sameNum == 6)  // 实际为检测出六帧才认为是成功检测
	{
		std::cout << "发现一辆黑烟车" << std::endl;
		isCatch = true;
		blackCarNum++;
		//sprintf(filename, "result/blackCar16-39-38不明显/%d.jpg", frameNum - 1);
		//imwrite(filename, src);
		sameNum++;
	}

	if (distance > src.rows / 4 || sameNum > 70)
	{
		sameNum = 0;
		rectCenters.clear();
		std::vector<Point2f>(rectCenters).swap(rectCenters);
	}
	rectCenters.push_back(newCenter);
	beforeFrameNum = frameNum;

	//waitKey(1);

}
