#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include <fstream>
#include <vector>
#include <iostream>
#include <string>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define linux
#ifdef linux
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif
#ifdef WIN
#include <direct.h>
#include <io.h>
#endif

using namespace std;
vector<string> getFiles(const string &cate_dir, bool append=false);
void getPointsFromString(string tempObject, vector<float> &points);
string int2string(int n, int i);
void getPointsFromTxt(string path, vector<float> &pose_points,vector<float> &face_points, 
	vector<float> &hand_left_points, vector<float> &hand_right_points);
