#include <string>
#include <vector>

using std::string;
using std::vector;

double information_gain (int n1, int n2, int p1, int p2);

template <class T>
class random_forest {
public:
    random_forest () = default;
    void read_num_trees (string&);
    void read_F (string&);
    void read_num_features (string&);
    void read_num_test (string&);
    void read_num_training (string&);
    void read_test_data (string&);
    void read_training_data (string&);
    void get_votes (vector<long>& test_votes,
                    vector<long>& outofbag_votes); // Paired true, false votes.
private:
    class dnode {
    public:
        dnode () = default;
        void set (random_forest*, int h, vector<int> training_indicies, dnode*);
        void set_child_dnodes (dnode*);
        bool transform_rowmajor (T*, int num_features) const; // Use on testing_data.
        bool transform_colmajor (T*, int num_training) const; // Use on training_data (transposed on input).
        bool transform_common (T) const;
    private:
        void transform_features_training (vector<int>& indicies, vector<T>& out) const;
        void transform_features_testing (vector<int>& indicies, vector<T>& out) const;
        random_forest* forest;
        int height_in_tree;
        vector<dnode*> child_dnodes;
        vector<T> splits;
        vector<double> split_cprobabilities;
        vector<bool> split_classes;
    };
private:
    int num_trees;
    int F;
    int num_features;
    int num_classes;
    int num_test;
    int num_training;
    int num_inbag;
    vector<T> test_data; 
    vector<T> training_data;
    vector<bool> class_data;
    vector<dnode> tree;
};

