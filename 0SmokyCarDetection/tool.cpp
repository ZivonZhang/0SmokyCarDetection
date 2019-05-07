#include "stdafx.h"
#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>

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


//×Ö·û´®·Ö¸îº¯Êý
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