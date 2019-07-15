#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>

using namespace std;

class Vertex {
public:
	Vertex() {
		vertex_id = -1;
		level = -1;
	}
	Vertex(int id) {
		this->vertex_id = id;
	}
	void set_id(int id) {
		this->vertex_id = id;
	}
	void set_level(int level) {
		this->level = level;
	}
	int get_id() {
		return vertex_id;
	}
	int get_level() {
		return level;
	}
private:
	int vertex_id;
	int level;
};

class Edge {
public:
	Edge() {
		this->edge_from = this->edge_to = NULL;
		this->edge_w = -1;
	}
	Edge(Vertex *edge_from, Vertex *edge_to, int edge_w=-1) {
		this->edge_from = edge_from;
		this->edge_to = edge_to;
		this->edge_w = edge_w;
	}
	Edge(Edge& e) {
		this->edge_from = e.get_edge_from();
		this->edge_to = e.get_edge_to();
		this->edge_w = e.get_edge_weight();
	}
	Vertex* get_edge_from() {
		return edge_from;
	}
	Vertex* get_edge_to() {
		return edge_to;
	}
	int get_edge_weight() {
		return edge_w;
	}
	void display() {
		printf("(%d--%d-->%d)",edge_from->get_id(),edge_w,edge_to->get_id());
	}
private:
	Vertex *edge_from, *edge_to;
	int edge_w;
};

class ListNode {
public:
	ListNode() {
		this->prev = this->next = NULL;
		this->is_a_head = false;
	}
	bool is_head() {
		return this->is_a_head;
	}
	ListNode* get_prev() {
		return this->prev;
	}
	ListNode* get_next() {
		return this->next;
	}
	void set_head(bool is_head) {
		this->is_a_head = is_head;
	}
	void set_prev(ListNode *prev) {
		this->prev = prev;
	}
	void set_next(ListNode *next) {
		this->next = next;
	}
	void insert_node(ListNode *pListNode) {
		ListNode *p_cur_node = this;
		while (!p_cur_node->is_head()) {
			assert(p_cur_node->get_prev());
			p_cur_node = p_cur_node->get_prev();
		}
		pListNode->set_prev(p_cur_node);
		pListNode->set_next(p_cur_node->next);
		if (p_cur_node->get_next() == NULL) {
			p_cur_node->set_next(pListNode);
		} else {
			p_cur_node->get_next()->set_prev(pListNode);
			p_cur_node->set_next(pListNode);
		}
	}
	void delete_node() {
		if (next) {
			next->set_prev(prev);
		} 
		prev->set_next(next);
		delete this;
	}
private:
	ListNode *prev, *next;
	bool is_a_head;
};

class VertexToEdgeListNode: public ListNode {
public:
	VertexToEdgeListNode() {
		this->edge = NULL;
		this->belong_to_vertex = NULL;
	}
	Edge* get_edge() {
		return this->edge;
	}
	void set_vertex(Vertex *belong_to_vertex) {
		this->belong_to_vertex = belong_to_vertex;
		this->set_head(true);
	}
	void set_edge(Edge *p_edge) {
		this->edge = p_edge;
	}
	void insert_edge(Edge *new_edge) {
		VertexToEdgeListNode *pListNode = new VertexToEdgeListNode;
		pListNode->set_edge(new_edge);
		this->insert_node(pListNode);
	}
	void delete_edge() {
		delete edge;
		delete_node();
	}
	void display_list() {
		VertexToEdgeListNode *p_cur_node;
		if (this->is_head()) {
			if (!this->get_next()) {
				printf("NULL\n");
				return;
			}
			else {
				p_cur_node = (VertexToEdgeListNode*)this->get_next();
			}
		}
		while (p_cur_node) {
			p_cur_node->get_edge()->display();
			printf(" ");
			p_cur_node = (VertexToEdgeListNode*)p_cur_node->get_next();
		}
		printf("\n");
	}
private:
	Edge *edge;
	Vertex *belong_to_vertex;
};


class Graph {
public:
	Vertex *vertex_pool;
	VertexToEdgeListNode *vertex_to_edge;
	
	Graph() {
		vertex_num = 0;
	}

	Graph(int vertex_num) {
		create_vertices(vertex_num);
	}
	~Graph() {
		for (int i = 0; i < vertex_num; i++) {
			;
		}
		delete [] vertex_to_edge;
		delete [] vertex_pool;
	}

	void create_vertices(int vertex_num) {
		assert(vertex_num >= 0);
		this->vertex_num = vertex_num;
		this->vertex_pool = new Vertex[vertex_num];
		this->vertex_to_edge = new VertexToEdgeListNode[vertex_num];
		for (int i = 0;i < vertex_num; i++) {
			this->vertex_pool[i].set_id(i);
			this->vertex_to_edge[i].set_vertex(&(this->vertex_pool[i]));
		}
	}

	void insert_edge(Vertex *from_vertex, Vertex *to_vertex, int edge_weight=-1) {
		Edge *pEdge = new Edge(from_vertex, to_vertex, edge_weight);
		vertex_to_edge[ from_vertex->get_id() ].insert_edge(pEdge);
	}

	void insert_edge(int from_vertex_id, int to_vertex_id, int edge_weight=-1) {
		this->insert_edge(&(this->vertex_pool[from_vertex_id]), &(this->vertex_pool[to_vertex_id]), edge_weight);
	}

	void insert_undirected_edge(int from_vertex_id, int to_vertex_id, int edge_weight=-1) {
		this->insert_edge(from_vertex_id, to_vertex_id, edge_weight);
		this->insert_edge(to_vertex_id, from_vertex_id, edge_weight);
	}

	void display_vertices() {
		for (int i = 0;i < vertex_num; i++) {
			printf("vertex_id: %d, ",vertex_pool[i].get_id());
			VertexToEdgeListNode *p_cur_node = &(vertex_to_edge[i]);
			while(p_cur_node->get_next()) {
				p_cur_node = (VertexToEdgeListNode*)p_cur_node->get_next();
				if (p_cur_node->get_edge()) {
					printf("%d->%d ", p_cur_node->get_edge()->get_edge_from()->get_id(), p_cur_node->get_edge()->get_edge_to()->get_id());
				}
			}
			printf("\n");
		}
	}

	void display() {
		for (int i = 0;i < vertex_num; i++) {
			printf("vertex_id: %d, ",vertex_pool[i].get_id());
			vertex_to_edge[i].display_list();
		}
	}

	void set_vertex_num(int vertex_num) {
		this->vertex_num = vertex_num;
	}
	int get_vertex_num() {
		return vertex_num;
	}
	int get_vertex_id(Vertex *vertex) {
		for (int i = 0; i < vertex_num; i++) {
			if (vertex == &(vertex_pool[i])) {
				return i;
			}
		}
		return -1;
	}

	VertexToEdgeListNode* get_edge_list(int vertex_id) {
		assert(vertex_id >= 0 && vertex_id < vertex_num);
		return &(vertex_to_edge[vertex_id]);
	}

	void compute_level(int root_id) {
		bool *has_visited = new bool[vertex_num];
		memset(has_visited, 0, sizeof(bool)*vertex_num);
		for (int i = 0;i < vertex_num; i++) {
			vertex_pool[i].set_level(-1);
		}
		queue<int> que;
		has_visited[root_id] = true;
		que.push(root_id);
		while(!que.empty()) {
			int u_id = que.front();
			que.pop();
			VertexToEdgeListNode *pListNode = (VertexToEdgeListNode*)vertex_to_edge[u_id].get_next();
			while (pListNode) {
				int v_id = pListNode->get_edge()->get_edge_to()->get_id();
				if (!has_visited[v_id]) {
					vertex_pool[v_id].set_level(vertex_pool[u_id].get_level() + 1);
					has_visited[v_id] = true;
					que.push(v_id);
				}
				pListNode = (VertexToEdgeListNode*)pListNode->get_next();
			}
		}
		delete [] has_visited;
	}
private:
	int vertex_num;
};

typedef int (*GetWeightFuncType)(vector<unsigned char>&, int, int);

class GridGraph: public Graph {
public:
	int grid_height, grid_width;
	GridGraph(int grid_height, int grid_width) {
		this->grid_height = grid_height;
		this->grid_width = grid_width;
		this->create_vertices(grid_height * grid_width);
		for (int i = 0;i < grid_height * grid_width; i++) {
			if (i - grid_width >= 0) {
				this->insert_edge(i, i - grid_width);
			}
			if (i % grid_width < grid_width - 1) {
				this->insert_edge(i, i + 1);
			}
			if (i + grid_width < grid_height * grid_width) {
				this->insert_edge(i, i + grid_width);
			}
			if (i % grid_width > 0) {
				this->insert_edge(i, i - 1);
			}
		}
	}
	GridGraph(int grid_height, int grid_width, vector<unsigned char>& image, GetWeightFuncType pfunc) {
		assert(grid_height * grid_width * 4 == image.size());
		this->grid_height = grid_height;
		this->grid_width = grid_width;
		this->create_vertices(grid_height * grid_width);
		for (int i = 0;i < grid_height * grid_width; i++) {
			if (i - grid_width >= 0) {
				this->insert_edge(i, i - grid_width, pfunc(image, i, i - grid_width));
			}
			if (i % grid_width < grid_width - 1) {
				this->insert_edge(i, i + 1, pfunc(image, i, i + 1));
			}
			if (i + grid_width < grid_height * grid_width) {
				this->insert_edge(i, i + grid_width, pfunc(image, i, i + grid_width));
			}
			if (i % grid_width > 0) {
				this->insert_edge(i, i - 1, pfunc(image, i, i - 1));
			}
		}
	}
	void display_vertices(int h_start, int h_end, int w_start, int w_end) {
		assert(h_start >= 0 && h_start < grid_height);
		assert(h_end >= 0 && h_end < grid_height);
		assert(w_start >= 0 && w_start < grid_width);
		assert(w_end >= 0 && w_end < grid_width);
		for (int h = h_start; h <= h_end; h++) {
			for (int w = w_start; w <= w_end; w++) {
				printf("[");
				printf("(%d,%d): ",h,w);
				VertexToEdgeListNode *p_cur_node = &(vertex_to_edge[h*grid_width+w]);
				while(p_cur_node->get_next()) {
					p_cur_node = (VertexToEdgeListNode*)p_cur_node->get_next();
					if (p_cur_node->get_edge()) {
						int to_id = p_cur_node->get_edge()->get_edge_to()->get_id();
						int e_w = p_cur_node->get_edge()->get_edge_weight();
						printf("(%d,%d,%d) ", to_id/grid_width, to_id%grid_width, e_w);
					}
				}
				printf("]\t");
			}
			printf("\n");
		}
	}
};