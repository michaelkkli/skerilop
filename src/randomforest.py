#! /usr/bin/python

from information_gain import information_gain
import numpy as np
import random

class RandomForest():
    def __init__(self, num_trees, F, training_set_indicies, features_array, features_class_array):
        self.training_set_indicies = training_set_indicies
        self.features_array = features_array
        self.features_class_array = features_class_array
        self.trees = []

        num_training_examples = len(training_set_indicies)
        num_inbag = int(num_training_examples*2./3.)
        num_outofbag = num_training_examples - num_inbag

        training_set_indicies_set = set(training_set_indicies)

        for _ in range(num_trees):
            outofbag_indicies = [
                training_set_indicies[random.randint(0,num_training_examples-1)] for _ in range(num_outofbag)]
            bag_indicies = [i for i in training_set_indicies_set.difference(outofbag_indicies)]
            self.trees.append(dtree(F, bag_indicies, features_array, features_class_array, outofbag_indicies))

    def summary(self):
        print(self.outofbag_stats())

    def outofbag_stats(self):
        votes_array_TF = np.zeros((self.features_array.shape[0], 2), dtype=np.int)

        for i in self.trees:
            i.outofbag_add_votes(votes_array_TF)

        num_diff = 0
        for i in self.training_set_indicies:
            forest_class = votes_array_TF[i, 0] > votes_array_TF[i, 1]
            if forest_class != False:
                num_diff += 1

        generalized_error = float(num_diff)/len(self.training_set_indicies)

        return {'generalized_error': generalized_error} 

class dtree():
    def __init__(self, F, training_set_indicies, features_array, features_class_array, obi):
        self.outofbag_indicies = obi
        self.training_set_indicies = training_set_indicies
        self.features_array = features_array
        self.features_class_array = features_class_array
        self.root_dnode = dnode(F-1, training_set_indicies, features_array, features_class_array)
        pass

    def generate_nodes(self):
        pass

    def outofbag_add_votes(self, votes_array_TF):
        """ Return out-of-bag indicies and classifications. """

        for i in self.outofbag_indicies:
            if self.root_dnode.transform(self.features_array[i,:]):
                votes_array_TF[i, 0] += 1
            else:
                votes_array_TF[i, 1] += 1

class dnode():
    def __init__(self, h, training_set_indicies, features_array, features_class_array):
        self.height_in_tree = h
        self.training_set_indicies = training_set_indicies

        self.feature_indicies = [] # Record-keeping.
        self.weights = None # None implies 1.0. Record-keeping.
        self.transform_features = None # Extract single feature or more complicated.

        self.split_vals = []
        self.child_nodes = []

        num_features = features_array.shape[1]
        self.features_indicies = [random.randrange(0, num_features)]

        if self.weights == None and len(self.features_indicies) == 1:
            self.transform_features = lambda x, c=self.features_indicies[0]: x[c]

        transformed_features = np.array(
            [self.transform_features(features_array[i,:]) for i in training_set_indicies])
        amin = np.amin(transformed_features)
        amax = np.amax(transformed_features)

        info_gain_best = 0.

        # Determine splits.
        for _ in range(10):
            split_val_try = np.random.uniform(amin, amax)

            left_inds = np.array(training_set_indicies)[transformed_features < split_val_try]
            right_inds = np.array(training_set_indicies)[transformed_features >= split_val_try]

            left_T = np.sum(features_class_array[left_inds])
            left_F = len(left_inds) - left_T

            right_T = np.sum(features_class_array[right_inds])
            right_F = len(right_inds) - right_T

            info_gain = information_gain(left_F, left_T, right_F, right_T)
            if info_gain > info_gain_best:
                info_gain_best = info_gain
                self.split_vals = [split_val_try]
            pass
        split_val_try = None

        left_inds = np.array(training_set_indicies)[transformed_features < self.split_vals[0]]
        right_inds = np.array(training_set_indicies)[transformed_features >= self.split_vals[0]]

        if self.height_in_tree > 0: # Create child dnodes.
            self.child_nodes = [
                dnodes(self.height_in_tree-1, left_inds, features_array, features_class_array),
                dnodes(self.height_in_tree-1, right_inds, features_array, features_class_array)]
        else: # Create classifications
            left_T = np.sum(features_class_array[left_inds])
            left_F = len(left_inds) - left_T

            right_T = np.sum(features_class_array[right_inds])
            right_F = len(right_inds) - right_T

            left_prob = float(left_T)/(left_T + left_F)
            right_prob = float(right_T)/(right_T + right_F)

            self.child_nodes = [
                lambda _, p=left_prob: True if random.random() < p else False,
                lambda _, p=right_prob: True if random.random() < p else False]

    def transform(self, input_features):
        """ Transform array of features and return classifications. """

        transformed_input_features = self.transform_features(input_features)
        for split, child in zip(self.split_vals, self.child_nodes):
            if transformed_input_features < split:
                if self.height_in_tree > 0:
                    return child.transform(input_features)
                else:
                    return child(None)

        if self.height_in_tree > 0:
            return self.child_nodes[-1].transform(input_features)
        else:
            return self.child_nodes[-1](None)

if __name__ == '__main__':
    use_wdbc = True

    if use_wdbc:
        with open('wdbc.data') as f:
            Y = np.genfromtxt(f, delimiter=',', usecols=(1), converters={1: lambda x: 'M' == x})
            f.seek(0)
            A = np.genfromtxt(f, delimiter=',', usecols=[i for i in range(2,32)])
            f.close()
        RF = RandomForest(200, 1, range(A.shape[0]), A, Y)
        RF.summary()
    else:
        with open('ntd.csv') as f:
            Y = np.genfromtxt(f, skip_header=1, delimiter=',', usecols=(21), converters={21: lambda x: '1' == x})
            f.seek(0)
            A = np.genfromtxt(f, skip_header=1, delimiter=',', usecols=range(0,21))
            f.close()
            print('Finished loading file.')
        RF = RandomForest(200, 1, range(A.shape[0]), A, Y)
        RF.summary()
