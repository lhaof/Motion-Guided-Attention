#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "lodepng.h"
#include "utils.h"

using namespace std;

class DataLoader {
public:
	string dataset, split;
	bool is_dense;
	vector<string> image_list, anno_list;
	int start_ix;
	DataLoader(const string& dataset, const string& split, bool is_dense) {
		assert(dataset == "DAVIS" || dataset == "FBMS");
		if (dataset == "DAVIS") {
			assert(split == "train" || split == "val");
		}
		if (dataset == "FBMS") {
			assert(split == "train" || split == "test");
		}
		this->dataset = dataset;
		this->split = split;
		this->is_dense = is_dense;
		string list_path;
		if (dataset == "DAVIS") {
			list_path = "/mnt/nvme/video_saliency/DAVIS/ImageSets/480p/" + split + ".txt";
		} else if (dataset == "FBMS") {
			list_path = "/mnt/nvme/video_saliency/FBMS/ImageSets/" + split;
			if (split == "test") {
				assert(is_dense = false);
			}
			if (is_dense) {
				list_path += "_dense";
			}
			list_path += ".txt";
		}
		ifstream infile(list_path.c_str());
		string line;
		vector<string> items;
		while (getline(infile, line)) {
			line = rstrip(line);
			items = splitstr(line);
			assert(items.size() == 1 || items.size() == 2);
			image_list.push_back(items[0]);
			if (items.size() == 2) {
				anno_list.push_back(items[1]);
			} else {
				anno_list.push_back(items[0].substr(0,items[0].length()-4)+".png");
			}
		}
		vector<string> templates;
		templates.push_back("/JPEGImages/480p");
		templates.push_back("/JPEGImages/Train");
		templates.push_back("/JPEGImages/Test");
		templates.push_back("/Annotations/480p");
		templates.push_back("/Annotations/Train");
		templates.push_back("/Annotations/Test");
		for (int i = 0; i < image_list.size(); i++) {
			for (int j = 0; j < templates.size(); j++) {
				if (startswith(image_list[i], templates[j])) {
					image_list[i] = image_list[i].substr(templates[j].length(), image_list[i].length() - templates[j].length());
					break;
				}
			}
		}
		if (anno_list.size()) {
			for (int i = 0; i < anno_list.size(); i++) {
				for (int j = 0; j < templates.size(); j++) {
					if (startswith(anno_list[i], templates[j])) {
						anno_list[i] = anno_list[i].substr(templates[j].length(), anno_list[i].length() - templates[j].length());
						cout << anno_list[i] << endl;
					}
				}
			}
		}
		start_ix = 0;
	}
	void get_images(const string& image_prefix, int batch_size, 
		vector< vector<unsigned char> >& images, 
		vector< pair<unsigned, unsigned> >& sizes) {
		assert(start_ix + batch_size <= image_list.size());
		images.clear();
		sizes.clear();
		for (int i = 0; i < batch_size; i++) {
			string image_name = image_prefix + image_list[start_ix + i];
			vector<unsigned char> image_buffer;
			unsigned height = 0, width = 0;
			unsigned _error = 0;
			if (endswith(image_name, string(".jpg"))) {
				_error = loadJpg(image_name.c_str(), height, width, image_buffer);
			} else {
				_error = lodepng::decode(image_buffer, width, height, image_name);
			}
			if (_error) {
				cout << "decoder error " << endl;
				cout << "image_name: " << image_name << endl;
				assert(false);
			}
			images.push_back(image_buffer);
			sizes.push_back(make_pair(height, width));
		}
	}
	void get_annos(const string& image_prefix, int batch_size, 
		vector< vector<unsigned char> >& images, 
		vector< pair<unsigned, unsigned> >& sizes) {
		assert(start_ix + batch_size <= anno_list.size());
		images.clear();
		sizes.clear();
		for (int i = 0; i < batch_size; i++) {
			string image_name = image_prefix + anno_list[start_ix + i];
			vector<unsigned char> image_buffer;
			unsigned height = 0, width = 0;
			unsigned _error = 0;
			if (endswith(image_name, string(".jpg"))) {
				_error = loadJpg(image_name.c_str(), height, width, image_buffer);
			} else {
				_error = lodepng::decode(image_buffer, width, height, image_name);
			}
			if (_error) {
				cout << "decoder error " << endl;
				cout << "image_name: " << image_name << endl;
				assert(false);
			}
			images.push_back(image_buffer);
			sizes.push_back(make_pair(height, width));
		}
	}
	void update_index(int batch_size) {
		start_ix += batch_size;
	}
};