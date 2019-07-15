#include <cstring>
#include <iostream>
#include <string>
#include <utility>

#include "dataloader.h"
#include "ADJ.h"
#include "utils.h"
#include "MST.h"

using namespace std;

const string DAVIS_IMAGE_DIR = "/mnt/nvme/video_saliency/DAVIS/JPEGImages/480p";
const string DAVIS_POS_DIR = "../compute_intersection/seeds_of_DAVIS/test-DAVIS-trainset/saliency_map";
const string DAVIS_NEG_DIR = "./seeds_of_DAVIS/test-DAVIS-trainset/MBD_map";

const string FBMS_IMAGE_DIR_TR = "/mnt/nvme/video_saliency/FBMS/JPEGImages/Train";
const string FBMS_IMAGE_DIR_TS = "/mnt/nvme/video_saliency/FBMS/JPEGImages/Test";
const string FBMS_POS_DIR = "../compute_intersection/seeds_of_FBMS/test-FBMS-trainset-dense/saliency_map";
const string FBMS_NEG_DIR = "./seeds_of_FBMS/test-FBMS-trainset-dense/MBD_map";

void solve(DataLoader *loader, const string& save_dir, int batch_size=1) {
	if (!if_directory_exist(save_dir.c_str())) {
		create_directory_ex(save_dir.c_str());
	}

	for (int i = 0; i < loader->image_list.size(); i++) {
		vector< vector<unsigned char> > images, annos;
		vector< pair<unsigned, unsigned> > image_sizes, anno_sizes;
		if (loader->dataset == "DAVIS") {
			loader->get_images(DAVIS_IMAGE_DIR, batch_size, images, image_sizes);
			loader->get_annos(DAVIS_POS_DIR, batch_size, annos, anno_sizes);
			loader->update_index(batch_size);
		} else if (loader->dataset == "FBMS") {
			if (loader->split == "train") {
				loader->get_images(FBMS_IMAGE_DIR_TR, batch_size, images, image_sizes);
				loader->get_annos(FBMS_POS_DIR, batch_size, annos, anno_sizes);
				loader->update_index(batch_size);
			} else {
				assert(false); // to-do
				loader->get_images(FBMS_IMAGE_DIR_TS, batch_size, images, image_sizes);
			}
		}
		// set seeds:
		// For each anno (pseudo), calculate the mean positive value as a threshold,
		// which is used to get shrinked positive seeds.
		for (int j = 0; j < images.size(); j++) {
			vector<unsigned char>& image_ = images[j];
			vector<unsigned char>& anno_ = annos[j];
			if (image_sizes[j] != anno_sizes[j]) {
				cout << "image_size: " << image_sizes[j].first << "*" << image_sizes[j].second << endl;
				cout << "anno_size: " << anno_sizes[j].first << "*" << anno_sizes[j].second << endl;
			}
			assert(image_sizes[j] == anno_sizes[j]);
			unsigned height = image_sizes[j].first;
			unsigned width = image_sizes[j].second;

			double mean_value = 0.;
			int pos_count = 0;
			assert(height*width*4 == anno_.size());
			for (int k = 0; k < height*width; k++) {
				if (anno_[k*4] > 0) {
					mean_value += (anno_[k*4] * 1.0);
					pos_count += 1;
				}
			}
			if (pos_count == 0) {
				mean_value = 256;
			} else {
				mean_value = mean_value / pos_count;
			}

			GridGraph graph(height, width, image_, &get_weight_func);
			MSTree mstree(&(graph.vertex_pool[0]), &graph);
			MBDMSTree mbdtree(&mstree, image_);
			for (int k = 0; k < height*width; k++) {
				if (anno_[k*4] > mean_value) {
					mbdtree.set_vertex_seed(k);
				}
			}
			mbdtree.compute_MBD();

			vector<unsigned char> salmap;
			for (int k = 0; k < height*width; k++) {
				for (int c = 0; c < 3; c++) {
					salmap.push_back((unsigned char)mbdtree.get_min_barrier_dist(k));
				}
				salmap.push_back((unsigned char)255);
			}

			string save_name = save_dir + loader->anno_list[i];
			int found = save_name.rfind("/");
			string last_dir = save_name.substr(0, found);
			if (!if_directory_exist(last_dir.c_str())) {
				create_directory_ex(last_dir.c_str());
			}
			unsigned _error = lodepng::encode(save_name, salmap, width, height);
			if (_error) {
				cout << "lodepng::encode error " << endl;
				cout << "save_name: " << save_name << endl;
				assert(false);
			}
		}
		if (i > 5) {
			cout << "i: " << i << endl;
			break;
		}
	}
}

int main() {
	bool davis_is_dense = true;
	DataLoader davis_loader("DAVIS", "train", davis_is_dense);
	cout << "DAVIS, num_of_images: " << davis_loader.image_list.size() << endl;
	cout << "DAVIS, num_of_annos: " << davis_loader.anno_list.size() << endl;

	bool fbms_is_dense = true;
	DataLoader fbms_loader("FBMS", "train", fbms_is_dense);
	cout << "FBMS, num_of_images: " << fbms_loader.image_list.size() << endl;
	cout << "FBMS, num_of_annos: " << fbms_loader.anno_list.size() << endl; 

	solve(&davis_loader, DAVIS_NEG_DIR);
	solve(&fbms_loader, FBMS_NEG_DIR);
	return 0;
}