#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <jpeglib.h>
#include <sys/stat.h>
#include <unistd.h> 

#ifndef _ADJ_H_
#define _ADJ_H_
#endif

using namespace std;

int get_weight_func(vector<unsigned char>& image, int i, int j);

string int_to_string(int x);

string rstrip(const string& str);

vector<string> splitstr(const string& str);

bool startswith(const string& str, const string& prefix);

bool endswith(const string& str, const string& posfix);

int loadJpg(const char* Name, unsigned &Height, unsigned &Width, vector<unsigned char>& Output);

bool if_directory_exist(const char *sPathName);

int create_directory_ex(const char *sPathName);