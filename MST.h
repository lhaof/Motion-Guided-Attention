#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <bitset>
#include <queue>
#include <utility>
#include <vector>

#ifndef _ADJ_H_
#define _ADJ_H_
#endif

#define MAX_WEIGHT 256

class WeightToEdgeListNode: public ListNode {
public:
	WeightToEdgeListNode() {
		this->edge = NULL;
		this->weight = -1;
	}
	Edge* get_edge() {
		return this->edge;
	}
	void set_weight(int weight) {
		this->weight = weight;
		this->set_head(true);
	}
	void set_edge(Edge *p_edge) {
		this->edge = p_edge;
	}
	void insert_edge(Edge *new_edge) {
		WeightToEdgeListNode *pListNode = new WeightToEdgeListNode;
		pListNode->set_edge(new_edge);
		this->insert_node(pListNode);
	}
	void display_list() {
		WeightToEdgeListNode *p_cur_node;
		if (this->is_head()) {
			if (!this->get_next()) {
				printf("NULL\n");
				return;
			}
			else {
				p_cur_node = (WeightToEdgeListNode*)this->get_next();
			}
		}
		while (p_cur_node) {
			p_cur_node->get_edge()->display();
			printf(" ");
			p_cur_node = (WeightToEdgeListNode*)p_cur_node->get_next();
		}
		printf("\n");
	}
private:
	Edge *edge;
	int weight;
};

class MSTree: public Graph {
public:
	MSTree() {
		;
	}
	MSTree(Vertex *root, Graph *graph) {
		this->root = root;
		this->set_vertex_num(graph->get_vertex_num());
		this->create_vertices(this->get_vertex_num());
		parent_edge = new Edge*[this->get_vertex_num()];
		for (int i = 0;i < this->get_vertex_num();i++) {
			parent_edge[i] = NULL;
		}
		has_chosen = new bool[this->get_vertex_num()];
		memset(has_chosen, 0, sizeof(bool)*this->get_vertex_num());
		verbose = false;
		int root_id = graph->get_vertex_id(root);
		assert(root_id >= 0 && root_id < this->get_vertex_num());
		prime_algorithm(root_id, graph);
	}
	void set_verbose(bool verbose=true) {
		this->verbose = verbose;
	}
	Vertex* get_root() {
		return root;
	}
	int get_parent_of(int vertex_id) {
		// use edge_from rather than edge_to
		// it is a little tricky, since the parent_edge is (parent --> child).
		if (!parent_edge[vertex_id]) {
			return -1;
		}
		return parent_edge[vertex_id]->get_edge_from()->get_id();
	}
	Edge* get_parent_edge(int vertex_id) {
		return parent_edge[vertex_id];
	}
private:
	Vertex *root;
	Edge **parent_edge;
	WeightToEdgeListNode *weight_to_edge;
	bitset<MAX_WEIGHT> weight_bset;
	bool *has_chosen;
	bool verbose;
	void prime_algorithm(int root_id, Graph *graph) {
		weight_to_edge = new WeightToEdgeListNode[weight_bset.size()];
		for (int i = 0; i < weight_bset.size(); i++) {
			weight_to_edge[i].set_weight(i);
		}
		expand_front(root_id, graph);
		if (verbose) {
			for (int i = 0; i < weight_bset.size(); i++) {
				if (weight_bset[i]) {
					weight_to_edge[i].display_list();
				}
			}
		}
		while(true) {
			unsigned firstOne = weight_bset._Find_first();
			if (firstOne >= weight_bset.size()) {
				break;
			}
			WeightToEdgeListNode *pListNode;
			pListNode = (WeightToEdgeListNode*)weight_to_edge[firstOne].get_next();
			assert(pListNode);
			int from_vertex_id = -1;
			int to_vertex_id = -1;
			// The following loop may be not necessary.
			while (true) {
				to_vertex_id = pListNode->get_edge()->get_edge_to()->get_id();
				if (!has_chosen[to_vertex_id] || !pListNode->get_next()) {
					break;
				} else {
					pListNode = (WeightToEdgeListNode*)pListNode->get_next();
					pListNode->get_prev()->delete_node();
				}
			}
			if (!has_chosen[to_vertex_id]) {
				from_vertex_id = pListNode->get_edge()->get_edge_from()->get_id();
				if (verbose) {
					printf("choose %d->%d\n", from_vertex_id, to_vertex_id);
				}
				vertex_to_edge[from_vertex_id].insert_edge(pListNode->get_edge());
				parent_edge[to_vertex_id] = pListNode->get_edge();
				expand_front(to_vertex_id, graph);
			}
			pListNode->delete_node();
			if (!weight_to_edge[firstOne].get_next()) {
				weight_bset[firstOne] = false;
			}
			if (verbose) {
				for (int i = 0; i < weight_bset.size(); i++) {
					if (weight_bset[i]) {
						weight_to_edge[i].display_list();
					}
				}
			}
		}
	}
	// We define 'front' as a set of edges whose start-vertex and end-vertex are from the growing MST and the non-chosen set respectively.
	void expand_front(int cur_vertex_id, Graph *graph) {
		has_chosen[cur_vertex_id] = true;
		VertexToEdgeListNode *edge_list_ptr = graph->get_edge_list(cur_vertex_id);
		while(edge_list_ptr->get_next()) {
			edge_list_ptr = (VertexToEdgeListNode*)edge_list_ptr->get_next();
			int vertex_id = edge_list_ptr->get_edge()->get_edge_to()->get_id();
			assert(vertex_id >= 0 && vertex_id < this->get_vertex_num());
			if (!has_chosen[vertex_id]) {
				int edge_weight = edge_list_ptr->get_edge()->get_edge_weight();
				weight_to_edge[edge_weight].insert_edge(edge_list_ptr->get_edge());
				weight_bset[edge_weight] = true;
			}
		}
	}
};

bool sortfunc(pair<int,int> a, pair<int,int> b) {
    return a.second > b.second;
}

class MBDMSTree: public MSTree {
public:
	MBDMSTree(MSTree *tree, vector<unsigned char>& image) {
		assert(tree->get_vertex_num()*4 == image.size());
		min_barrier_dist = new int[tree->get_vertex_num()];
		is_seed = new bool[tree->get_vertex_num()];
		max_value_along_path = new int[tree->get_vertex_num()*3];
		min_value_along_path = new int[tree->get_vertex_num()*3];
		vertex_value = new int[tree->get_vertex_num()*3];
		for (int i = 0;i < tree->get_vertex_num();i++) {
			min_barrier_dist[i] = -1;
			is_seed[i] = false;
			for (int j = 0;j < 3;j++) {
				max_value_along_path[i*3+j] = image[i*4+j];
				min_value_along_path[i*3+j] = image[i*4+j];
				vertex_value[i*3+j] = image[i*4+j];
			}
		}
		this->tree = tree;
	}
	void compute_MBD() {
		int seed_count = 0;
		for (int i = 0;i < tree->get_vertex_num();i++) {
			if (is_seed[i]) {
				min_barrier_dist[i] = 0;
				seed_count++;
			}
		}
		if (seed_count == 0) {
			throw "ERROR: There is no seeds!\n";
		}
		if (seed_count == tree->get_vertex_num()) {
			throw "ERROR: All vertices are seeds!\n";
		}
		bottom_up();
		top_down();
	}
	void display_MBD() {
		for (int i = 0;i < tree->get_vertex_num();i++) {
			printf("vertex_id: %d min_barrier_dist[%d]: %d\n",i,i,min_barrier_dist[i]);
		}
	}
	void set_vertex_seed(int vertex_id, bool is_seed=true) {
		this->is_seed[vertex_id] = is_seed;
	}
	int get_min_barrier_dist(int vertex_id) {
		return min_barrier_dist[vertex_id];
	}
private:
	MSTree *tree;
	int *min_barrier_dist;
	bool *is_seed;
	int *max_value_along_path;
	int *min_value_along_path;
	int *vertex_value;
	void bottom_up() {
		tree->compute_level(tree->get_root()->get_id());
		vector< pair<int, int> > vec;
		for (int i = 0;i < tree->get_vertex_num();i++) {
			vec.push_back(make_pair(i, tree->vertex_pool[i].get_level()));
		}
		sort(vec.begin(), vec.end(), sortfunc);
		for (int i = 0;i < vec.size();i++) {
			int v_id = vec[i].first;
			int u_id = tree->get_parent_of(v_id);
			if (u_id == -1) {
				// v has not parent, which implies v is the root.
				break;
			}
			if (min_barrier_dist[v_id] != -1)  {
				int tmp_dist, tmp_min[3], tmp_max[3];
				for (int j = 0;j < 3;j++) {
					tmp_min[j] = min(vertex_value[u_id*3+j], min_value_along_path[v_id*3+j]);
					tmp_max[j] = max(vertex_value[u_id*3+j], max_value_along_path[v_id*3+j]);
					tmp_dist = (j == 0) ? (tmp_max[j] - tmp_min[j]) : min(tmp_dist, tmp_max[j] - tmp_min[j]);
				}
				if (min_barrier_dist[u_id] == -1 || tmp_dist < min_barrier_dist[u_id]) {
					min_barrier_dist[u_id] = tmp_dist;
					for (int j = 0;j < 3;j++) {
						min_value_along_path[u_id*3+j] = tmp_min[j];
						max_value_along_path[u_id*3+j] = tmp_max[j];
					}
				}
			}
		}
	}
	void top_down() {
		queue<int> que;
		bool *inque = new bool[tree->get_vertex_num()];
		memset(inque, 0, sizeof(bool)*tree->get_vertex_num());
		int root_id = tree->get_root()->get_id();
		que.push(root_id);
		inque[root_id] = true;
		assert(min_barrier_dist[root_id] != -1);
		while (!que.empty()) {
			int v_id = que.front();
			que.pop();
			VertexToEdgeListNode *pListNode = (VertexToEdgeListNode*)tree->vertex_to_edge[v_id].get_next();
			while (pListNode) {
				int u_id = pListNode->get_edge()->get_edge_to()->get_id();
				if (min_barrier_dist[v_id] != -1) {
					int tmp_dist, tmp_min[3], tmp_max[3];
					for (int j = 0;j < 3;j++) {
						tmp_min[j] = min(vertex_value[u_id*3+j], min_value_along_path[v_id*3+j]);
						tmp_max[j] = max(vertex_value[u_id*3+j], max_value_along_path[v_id*3+j]);
						tmp_dist = (j == 0) ? (tmp_max[j] - tmp_min[j]) : min(tmp_dist, tmp_max[j] - tmp_min[j]);
					}
					if (min_barrier_dist[u_id] == -1 || tmp_dist < min_barrier_dist[u_id]) {
						min_barrier_dist[u_id] = tmp_dist;
						for (int j = 0;j < 3;j++) {
							min_value_along_path[u_id*3+j] = tmp_min[j];
							max_value_along_path[u_id*3+j] = tmp_max[j];
						}
					}
					if (!inque[u_id]) {
						que.push(u_id);
						inque[u_id] = true;
					}
				}
				pListNode = (VertexToEdgeListNode*)pListNode->get_next();
			}
		}
	}
};


