#! /usr/bin/python
import numpy as np
import random
from pprint import pprint

class RandomForest():

    def __init__(self):
        self.forest_training_data = None
        self.forest_classcat = None
        self.num_examples = 0
        self.feature_split_less = []
        self.feature_split_greater = []
        self.bag_index_count_less = []
        self.bag_index_count_greater = []

    def reset_bagging(self, num_trees, bag_size, examples, classcat):
        if num_trees <= 0:
            raise ValueError('num_trees must be greater than zero.')

        if bag_size <= 0:
            raise ValueError('bag_size must be greater than zero.')

        num_examples = examples.shape[0]
        if bag_size > num_examples:
            raise ValueError('bag_size is too large.')
        self.num_examples = num_examples

        if len(classcat) != num_examples:
            raise ValueError('Number of rows in examples does not match length of classcat.')

        self.num_trees = num_trees
        self.forest_training_data = examples
        self.forest_classcat = classcat

        num_features = examples.shape[1]
        half_num_examples = .5 * num_examples

        self.feature_split_less = []
        self.feature_split_greater = []
        self.bag_index_count_less = []
        self.bag_index_count_greater = []

        for _ in range(num_trees):
            bag_index_count = [0]*self.num_examples

            for _ in range(bag_size):
                bag_index_count[random.randint(0, self.num_examples-1)] += 1

            btmp1 = []
            btmp2 = []

            for i, j in enumerate(bag_index_count):
                if j > 0:
                    btmp1.extend([self.forest_training_data[i, :] for _ in range(j)])
                    btmp2.extend([self.forest_classcat[i] for _ in range(j)])

            bag = np.array(btmp1)
            bag_classcat = np.array(btmp2)

            col = random.randrange(0, num_features)

            median = np.median(bag[:, col])

            num_less = np.sum(bag_classcat[i] for i in range(bag_size) if bag[i, col] < median)

            if (num_less < half_num_examples):
                self.feature_split_greater.append((col, median))
                self.bag_index_count_greater.append(bag_index_count)
            else:
                self.feature_split_less.append((col, median))
                self.bag_index_count_less.append(bag_index_count)

    def transform(self, features_it):
        for fit in features_it:
            votes_less = np.sum(fit[i] < j for i, j in feature_split_less)
            votes_greater = np.sum(fit[i] >= j for i, j in feature_split_greater)

            all_votes = votes_less + votes_greater
            percentage_votes = all_votes/self.num_trees

            if percentage_votes >= 0.5:
                yield True, percentage_votes
            else:
                yield False, 1. - percentage_votes

    def out_of_bag_stats(self):
        classcat = [False]*self.num_examples
        strength_numer = 0.
        sum_squares = 0.
        sd_numer = 0.

        for eg in range(self.num_examples):
            votes_less = 0
            votes_greater = 0
            max_outofbag_votes = 0

            for ft_sp, ind_cnt in zip(self.feature_split_less, self.bag_index_count_less):
                if ind_cnt[eg] == 0: #If out of bag
                    max_outofbag_votes += 1
                    if self.forest_training_data[eg, ft_sp[0]] < ft_sp[1]:
                        votes_less += 1
            for ft_sp, ind_cnt in zip(self.feature_split_greater, self.bag_index_count_greater):
                if ind_cnt[eg] == 0: #If out of bag
                    max_outofbag_votes += 1
                    if self.forest_training_data[eg, ft_sp[0]] >= ft_sp[1]:
                        votes_greater += 1

            all_votes = votes_less + votes_greater
            percentage_votes = float(all_votes)/max_outofbag_votes

            if percentage_votes >= 0.5: #If forest votes for true
                classcat[eg] = True
                if self.forest_classcat[eg] == True: #If right
                    tmp = 2.*percentage_votes - 1.
                    p1 = percentage_votes
                    p2 = 1. - percentage_votes
                else: #Else wrong
                    tmp = 1. - 2*percentage_votes
                    p1 = 1. - percentage_votes
                    p2 = percentage_votes
            else: #Else forest votes for False
                if self.forest_classcat[eg] == False: #If right
                    tmp = 1. - 2*percentage_votes
                    p1 = 1. - percentage_votes
                    p2 = percentage_votes
                else: #Else wrong
                    tmp = 2.*percentage_votes - 1.
                    p1 = percentage_votes
                    p2 = 1. - percentage_votes

            sum_squares += tmp*tmp
            strength_numer += tmp

            p1mp2 = p1 - p2
            sd_numer += np.sqrt(p1 + p2 + p1mp2*p1mp2)

        strength_s = strength_numer/self.num_examples

        var_mr = sum_squares/self.num_examples - strength_s*strength_s
        Esd = sd_numer/self.num_examples
        Esd_squared = Esd*Esd
        correlation_rhohat = var_mr/Esd_squared

        num_diffs = np.sum([i != j for i, j in zip(classcat, self.forest_classcat)])
        return { 'generalized_error' : float(num_diffs)/len(classcat),
                 'strength' : strength_s,
                 'correlation' : correlation_rhohat }
            
if __name__ == '__main__':
    RF = RandomForest()

    with open('wdbc.data') as f:
        Y = np.genfromtxt(f, delimiter=',', usecols=(1), converters={1: lambda x: 'M' == x})
        f.seek(0)
        A = np.genfromtxt(f, delimiter=',', usecols=[i for i in range(0,32) if i != 1],
                          converters={1: lambda x: 'M' == x})
        f.close()

    RF.reset_bagging(400, int(A.shape[0]*2./3), A, Y)
    print(RF.out_of_bag_stats())
