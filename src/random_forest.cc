#include "random_forest.hh"

#include <algorithm>
#include <fstream>
#include <string>

using std::ifstream;
using std::maximum_element;
using std::minimum_element;
using std::string;

template <class T>
void
random_forest<T>::read_num_trees (std::string& in)
{
    ifstream f(in);
    f >> this->num_trees;
}

template <class T>
void
random_forest<T>::read_F (std::string& in)
{
    ifstream f(in);
    f >> this->F;
    this->tree.resize ( static_cast<unsigned>(1)<<(this->F) - 1); // 2^{F}-1 (geometric series sum).
}

template <class T>
void
random_forest<T>::read_num_features (std::string& in)
{
    ifstream f(in);
    f >> this->num_features;
}

template <class T>
void
random_forest<T>::read_num_test (std::string& in)
{
    ifstream f(in);
    f >> this->num_test;
}

template <class T>
void
random_forest<T>::read_num_training (std::string& in)
{
    ifstream f(in);
    f >> this->num_training;
}

template <class T>
void
random_forest<T>::read_test_data (std::string& in)
{
    ifstream f(in);

    int len = num_test * num_features;
    this->test_data.resize (len);

    T* it = &test_data[0];

    for (int i = 0; i < len; ++i) {
        f >> *(it++);
    }
}

template <class T>
void
random_forest<T>::read_training_data (std::string& in)
{
    bool tmp;

    ifstream f(in);

    this->training_data.resize (num_training * (num_features + 1));
    this->class_data.resize (num_training);

    T* it = &training_data[0];
    vector<bool>::iterator cit = class_data.begin();

    for (int i = 0; i < num_training; ++i) {
        for (int j = 0; j < num_features; ++j) {
            f >> *(it++);
        }
        f >> tmp;
        *(cit++) = tmp;
    }
}

template <class T>
void
random_forest<T>::get_votes (vector<long>& test_votes, vector<long>& outofbag_votes)
{
}

template <class T>
void
random_forest<T>::dnode::set (random_forest* rf, int h, vector<int>& training_ind, dnode** child)
{
    this->forest = rf;
    this->height_in_tree = h;
    vector<int> left_ind, right_ind;
    vector<T> transformed_features (training_ind.size ());

    if (training_ind.size() < 11) {
        this->height_in_tree = 0;
    }

    this->transform_features_training (training_ind, transformed_features);

    int len = training_ind.size ();
    T amin = *(minimum_element (transformed_features.begin (), transformed_features.end ()));
    T amax = *(maximum_element (transformed_features.begin (), transformed_features.end ())); 

    random_device rd;

    double info_gain_best = 0.;
    T split_val_try;
    int left_F, left_T, right_F, right_T;

    for (int trial = 0; trial < 10; ++trial) {
        split_val_try = static_cast<T> (amin + (double(rd()) / random_device::max) * (amax - amin));

        left_ind.clear (); right_ind.clear ();

        for (int i = 0; i < len; ++i) {
            if (transformed_features[i] < split_val_try) {
                left_ind.push_back (i);
            } else {
                right_ind.push_back (i);
            }
        }

        left_T = 0;
        for (j : left_ind) { if (this->forest->class_data[j]) ++left_T; }
        left_F = left_ind.size () - left_T;

        right_T = 0;
        for (j : right_ind) { if (this->forest->class_data[j]) ++right_T; }
        right_F = right_ind.size () - right_T;

        double info_gain = information_gain (left_F, left_T, right_F, right_T);
        if (info_gain > info_gain_best) {
            info_gain_best = info_gain;

            this->splits.resize (1);
            this->splits [0] = split_val_try;
        }
    }

    left_ind.clear (); right_ind.clear ();

    split_val_try = this->splits[0];

    for (int i = 0; i < len; ++i) {
        if (transformed_features[i] < split_val_try) {
            left_ind.push_back (i);
        } else {
            right_ind.push_back (i);
        }
    }

    if (0 == left_ind.size () || 0 == right_ind.size ()) {
        this->height_in_tree = 0;
    }

    left_T = 0;
    for (j : left_ind) { if (this->forest->class_data[j]) ++left_T; }

    right_T = 0;
    for (j : right_ind) { if (this->forest->class_data[j]) ++right_T; }

    if (height_in_tree > 0) {
        left_F = left_ind.size () - left_T;
        right_F = right_ind.size () - right_T;

        double left_prob, right_prob;
        if (left_ind.size () > 0 && right_ind.size () > 0) {
            left_prob = static_cast<double>(left_T)/(left_T + left_F);
            right_prob = static_cast<double>(right_T)/(right_T + right_F);
        } else if (0 == left_ind.size ()) {
            left_prob = 0.01; // Disallow probability zero.
            right_prob = 0.99; // Disallow probability one.
        } else { // right_ind.size () must be zero.
            left_prob = 0.99; // Disallow probability zero.
            right_prob = 0.01; // Disallow probability one.
        }

        this->split_cprobabilities.resize (1);
        if (left_prob < right_prob) {
            this->split_cprobabilities[0] = right_prob;
        } else {
            this->split_cprobabilities[0] = left_prob;
        }
        child_dnodes[0] = (*child)++;
        child_dnodes[1] = (*child)++;

        child_dnodes[0]->set(rf, h-1, left_ind, child);
        child_dnodes[1]->set(rf, h-1, right_ind, child);
    } else {
        int left_T_right_T = left_T + right_T;

        double T_prob, F_prob;
        if (left_T_right_T > 0 && left_T_right_T < len) {
            T_prob = static_cast<double>(left_T_right_T)/len;
            F_prob = 1. - T_prob;
            this->split_cprobabilities.resize (1);
            this->split_classes.resize (2);
            if (T_prob < F_prob) {
                this->split_cprobabilities[0] = F_prob;
                this->split_classes[0] = false;
                this->split_classes[1] = true;
            } else {
                this->split_cprobabilities[0] = T_prob;
                this->split_classes[0] = true;
                this->split_classes[1] = false;
            }
        } else if (0 == left_T_right_T) {
            T_prob = 0.01;
            F_prob = 0.99;
        } else { // left_T_right_T must be len.
            T_prob = 0.99;
            F_prob = 0.01;
        }
    }
}

template <class T>
bool
random_forest<T>::dnode::transform_rowmajor (T*, int num_features) const
{
}

template <class T>
bool
random_forest<T>::dnode::transform_colmajor (T*, int num_training) const
{
}

template <class T>
bool
random_forest<T>::dnode::transform (T in) const
{
}


template class random_forest<int>;
