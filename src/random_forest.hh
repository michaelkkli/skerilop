#include <vector>

class std::string;

double information_gain (int n1, int n2, int p1, int p2);

template <class T>
class random_forest {
public:
    random_forest ();
    void read_num_trees (std::string&);
    void read_F (std::string&);
    void read_test_data (std::string&);
    void read_training_data (std::string&);
    void get_votes (std::vector<long> test_votes,
                    std::vector<long> outofbag_votes); // Paired true, false votes.
private:
    int num_trees;
    int F;
    int num_features;
    int num_classes;
    int num_test;
    int num_training;
    int num_inbag;
    std::vector<T> test_data; 
    std::vector<T> training_data;
    std::vector<bool> class_data;
    std::vector<dnode>;
private:
    class dnode {
    public:
        dnode ();
        void set (random_forest*, int h);
        void set_child_dnodes (dnode*);
        bool transform_rowmajor (T*, int num_features) const; // Use on testing_data.
        bool transform_colmajor (T*, int num_training) const; // Use on training_data (transposed on input).
        bool transform_common (T) const;
    private:
        random_forest* forest;
        int height_in_tree;
        std::vector<dnode*> child_dnodes;
        std::vector<T> splits;
        std::vector<double> split_cprobabilities;
        std::vector<bool> split_classes;
    };
};

