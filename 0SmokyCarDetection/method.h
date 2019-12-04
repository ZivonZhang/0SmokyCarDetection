#ifndef METHOD_H
#define METHOD_H


//#include "stdafx.h"

//using namespace std;
using namespace cv;

class Method
{
public:
	Method(std::string R);
	~Method(){};
	 

	void judgeSomkeCars(Mat &src, Rect &rect, int num, bool &isCatch);

	int getBlackCarNum(){ return blackCarNum; };
	
public:
	int out_count = 0;  //����ĳ�β��ͼƬ�ı��
	char file_name[100];
	int trunkNum = 0;
	int ContinuityThreshold = 8;
	std::queue<Mat> area_Smoke;
	std::string nowFolderPath;

private:
	void clear(std::queue<Mat>& q);

	std::string ResultPath;

	int count = 0; // ��¼û�����仯��ͼ���֡��

	std::vector<Point2f> rectCenters;  //  ��¼ÿ����⵽�ĺ��̳��Ŀ�����ĵ�
	int sameNum = 0;  //  ��¼����ͬһ�����̳��ĸ���
	int blackCarNum = 0;  //  ʶ������̳�����
	int beforeFrameNum = 0;  // �ϴμ�⵽���̳���֡��
	char filename[100];  // ͼƬ����·��
};

#endif