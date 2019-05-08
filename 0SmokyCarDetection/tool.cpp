#include "stdafx.h"
#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>
#include <io.h>
#include <direct.h>

using namespace cv;

std::string findFile(const std::string& filename)
{
	if (filename.empty() || utils::fs::exists(filename))
		return filename;

	const char* extraPaths[] = { getenv("OPENCV_DNN_TEST_DATA_PATH"),
		getenv("OPENCV_TEST_DATA_PATH") };
	for (int i = 0; i < 2; ++i)
	{
		if (extraPaths[i] == NULL)
			continue;
		std::string absPath = utils::fs::join(extraPaths[i], utils::fs::join("dnn", filename));
		if (utils::fs::exists(absPath))
			return absPath;
	}
	CV_Error(Error::StsObjectNotFound, "File " + filename + " not found! "
		"Please specify a path to /opencv_extra/testdata in OPENCV_DNN_TEST_DATA_PATH "
		"environment variable or pass a full path to model.");
}

void getFiles(std::string path, std::vector<std::string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

//新建多级文件夹
void CreateDir(const char *dir)
{
	int m = 0, n;
	std::string str1, str2;

	str1 = dir;
	str2 = str1.substr(0, 2);
	str1 = str1.substr(3, str1.size());

	while (m >= 0)
	{
		m = str1.find('/');

		str2 += '/' + str1.substr(0, m);
		n = _access(str2.c_str(), 0); //判断该目录是否存在
		if (n == -1)
		{
			_mkdir(str2.c_str());     //创建目录
		}
		str1 = str1.substr(m + 1, str1.size());
	}
}

//字符串分割函数
std::vector< std::string> split(std::string str, std::string pattern)
{
	std::vector<std::string> ret;
	if (pattern.empty()) return ret;
	size_t start = 0, index = str.find_first_of(pattern, 0);
	while (index != str.npos)
	{
		if (start != index)
			ret.push_back(str.substr(start, index - start));
		start = index + 1;
		index = str.find_first_of(pattern, start);
	}
	if (!str.substr(start).empty())
		ret.push_back(str.substr(start));
	return ret;
}

void getTime()  //系统时间
{
	struct tm t;   //tm结构指针
	time_t now;  //声明time_t类型变量
	time(&now);      //获取系统日期和时间
	localtime_s(&t, &now);   //获取当地日期和时间

	printf("%d/", t.tm_year + 1900);
	printf("%d/", t.tm_mon + 1);
	printf("%d ", t.tm_mday);
	printf("%d:", t.tm_hour);
	printf("%d:", t.tm_min);
	printf("%d\n", t.tm_sec);
}