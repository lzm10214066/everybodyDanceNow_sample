#include "utility.h"

vector<string> getFiles(const string &cate_dir, bool append)
{
	vector<string> files;//存放文件名  
	string tmp_cate_dir = cate_dir + "\\*";
#ifdef WIN  
	struct __finddata64_t file;
	__int64 lf;
	//输入文件夹路径  
	if ((lf = _findfirst64(tmp_cate_dir.c_str(), &file)) == -1) {
		cout << cate_dir << " not found!!!" << endl;
		exit(-1);
	}
	else {
		while (_findnext64(lf, &file) == 0) {
			//输出文件名  
			//cout<<file.name<<endl;  
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
				continue;
			if (append) files.push_back(cate_dir + "/" + file.name);
			else files.push_back(file.name);
		}
	}
	_findclose(lf);
#endif  

#ifdef linux  
	DIR *dir;
	struct dirent *ptr;
	char base[1000];

	if ((dir = opendir(cate_dir.c_str())) == NULL)
	{
		perror("Open dir error...");
		cout << cate_dir << endl;
		exit(1);
	}
	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir  
			continue;
		else if (ptr->d_type == 8)    ///file  
		{
			if (append) files.push_back(cate_dir + "/" + ptr->d_name);
			else files.push_back(ptr->d_name);
		}
		else if (ptr->d_type == 10)    ///link file  
			continue;
		else if (ptr->d_type == 4)    ///dir  
		{
			if (append) files.push_back(cate_dir + "/" + ptr->d_name);
			else files.push_back(ptr->d_name);
		}
	}
	closedir(dir);
#endif 
	sort(files.begin(), files.end());
	return files;
}


void getPointsFromString(string tempObject, vector<float> &points)
{
	vector<int> posi;
	posi.push_back(-1);
	for (int i = 0; i != tempObject.size(); ++i)
	{
		if (tempObject[i] == ',')
		{
			posi.push_back(i);
		}
	}
	posi.push_back(tempObject.size());

	for (int i=0;i<posi.size()-1;++i)
	{
		points.push_back(atof(tempObject.substr(posi[i] + 1, posi[i+ 1] - posi[i] - 1).c_str()));
	}
}


void getPointsFromTxt(string path, vector<float> &pose_points,
	vector<float> &face_points, vector<float> &hand_left_points, vector<float> &hand_right_points) {

	string buf;
	ifstream points_list(path);
	if (!points_list) {
		cout << "cannot open the file:" << path.c_str() << endl;
	}

	vector<string> bufs;
	while (points_list) {
		if (getline(points_list, buf)) {
			bufs.push_back(buf);
		}
	}
	getPointsFromString(bufs[0], pose_points);
	getPointsFromString(bufs[1], face_points);
	getPointsFromString(bufs[2], hand_left_points);
	getPointsFromString(bufs[3], hand_right_points);

}

string int2string(int n, int i)
{
	string s = std::to_string(i);
	int len =s.size();
	if (len > n)
	{
		cout << "输入的N太小！";
		return string();
	}
	else
	{
		stringstream Num;
		for (int i = 0; i < n - len; i++)
			Num << "0";
		Num << i;

		return Num.str();
	}
}
