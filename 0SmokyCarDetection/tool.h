#include "stdafx.h"
#pragma once
#ifndef tool_H
#define tool_H
#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;

std::string findFile(const std::string& filename);
std::vector< std::string> split(std::string str, std::string pattern);
void getFiles(std::string path, std::vector<std::string>& files);
void getTime();  //系统时间
void CreateDir(const char *dir);


#endif
