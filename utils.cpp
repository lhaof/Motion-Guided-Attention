#include "utils.h"

int get_weight_func(vector<unsigned char>& image, int i, int j) {
	assert(i >= 0 && i * 4 + 3 < image.size());
	assert(j >= 0 && j * 4 + 3 < image.size());
	return max(abs(image[i*4]-image[j*4]), max(abs(image[i*4+1]-image[j*4+1]), abs(image[i*4+2]-image[j*4+2])));
}

string int_to_string(int x) {
	assert(x >= 0);
	string str = "";
	while (x) {
		str = char((x % 10) + '0') + str;
		x /= 10;
	}
	if (str == "") {
		str = "0";
	}
	return str;
}

string rstrip(const string& str) {
	string out = "";
	if (str.length() == 0) {
		return out;
	}
	int idx = str.length()-1;
	while (str[idx] == '\n' || str[idx] == '\r' || str[idx] == ' ') {
		idx--;
	}
	if (idx < 0) {
		return out;
	}
	return str.substr(0,idx+1);
}

vector<string> splitstr(const string& str) {
	vector<string> vec;
	int slen = str.length();
	int start_ix = 0;
	int end_ix = 0;
	while (start_ix < slen && end_ix < slen) {
		if (str[end_ix] == ' ') {
			if (end_ix - start_ix >= 1) {
				vec.push_back(str.substr(start_ix, end_ix - start_ix));
			}
			start_ix = end_ix + 1;
		} else if (end_ix == slen-1) {
			vec.push_back(str.substr(start_ix, end_ix - start_ix + 1));
		}
		end_ix = end_ix + 1;
	}
	return vec;
}

bool startswith(const string& str, const string& prefix) {
	if (prefix.length() > str.length()) {
		return false;
	}
	for (int i = 0;i < prefix.length();i++) {
		if (str[i] != prefix[i]) {
			return false;
		}
	}
	return true;
}

bool endswith(const string& str, const string& posfix) {
	if (posfix.length() > str.length()) {
		return false;
	}
	int i = str.length() - 1;
	int j = posfix.length() - 1;
	while (j >= 0) {
		if (str[i] != posfix[j]) {
			return false;
		}
		i--;
		j--;
	}
	return true;
}


int loadJpg(const char* Name, unsigned &Height, unsigned &Width, vector<unsigned char>& Output) {
	unsigned char a, r, g, b;
	int width, height;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	FILE * infile;        /* source file */
	JSAMPARRAY pJpegBuffer;       /* Output row buffer */
	int row_stride;       /* physical row width in output buffer */
	if ((infile = fopen(Name, "rb")) == NULL) {
		fprintf(stderr, "can't open %s\n", Name);
		return 0;
	}
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);
	(void) jpeg_read_header(&cinfo, TRUE);
	(void) jpeg_start_decompress(&cinfo);
	width = cinfo.output_width;
	height = cinfo.output_height;

	Output.clear();
	/*
	unsigned char * pDummy = new unsigned char [width*height*4];
	unsigned char * pTest = pDummy;
	if (!pDummy) {
		printf("NO MEM FOR JPEG CONVERT!\n");
		return 0;
	}
	*/
	row_stride = width * cinfo.output_components;
	pJpegBuffer = (*cinfo.mem->alloc_sarray)
	((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

	while (cinfo.output_scanline < cinfo.output_height) {
		(void) jpeg_read_scanlines(&cinfo, pJpegBuffer, 1);
		for (int x = 0; x < width; x++) {
			a = 0; // alpha value is not supported on jpg
			r = pJpegBuffer[0][cinfo.output_components * x];
			if (cinfo.output_components > 2) {
				g = pJpegBuffer[0][cinfo.output_components * x + 1];
				b = pJpegBuffer[0][cinfo.output_components * x + 2];
			} else {
				g = r;
				b = r;
			}
			/*
			*(pDummy++) = b;
			*(pDummy++) = g;
			*(pDummy++) = r;
			*(pDummy++) = a;
			*/
			Output.push_back(r);
			Output.push_back(g);
			Output.push_back(b);
			Output.push_back(a);
		}
	}
	fclose(infile);
	(void) jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	// BMap = (int*)pTest; 
	Height = (unsigned)height;
	Width = (unsigned)width;
	// Depth = 32;
	return 0;
}

bool if_directory_exist(const char *sPathName) {
	int a = access(sPathName, F_OK);
	if (a == -1) {
		return false;
	}
	return true;
}

int create_directory_ex(const char *sPathName) {
	char DirName[256];
	strcpy(DirName,sPathName);
	int len = strlen(DirName);
	if (DirName[len-1] != '/') {
		strcat(DirName,"/");
	}
	len = strlen(DirName);
	for (int i = 1;i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			int a = access(DirName, F_OK);
			if(a == -1) {
				mkdir(DirName, 0755);
			}
			DirName[i] = '/';
		}
	}
	return 0;
}