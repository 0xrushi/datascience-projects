# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from typing import Iterable

class Node:
    def __init__(self, n_samples, ddp, ate):
        self.n_samples = n_samples
        self.left = None
        self.right = None
        # delta delta p
        self.ddp = ddp
        # average treatment effect
        self.ate = ate
        self.split_feature = None
        self.split_threshold = None

class UpliftTreeRegressor:
    def __init__(
       self,
       max_depth: int =3,  # max tree depth
       min_samples_leaf: int = 6000, # min number of values in leaf
       min_samples_leaf_treated: int = 2500, # min number of treatment values in leaf
       min_samples_leaf_control: int = 2500,  # min number of control values in leaf
       ):

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        
    def best_split(self, x, treatment, y) -> (int, float):
        best_ddp = 0.0
        split_feat = None
        split_thres = None
        for feat_n in range(self.n_features):
            # get threshold options
            thres_op = list(self.threshold_options(x[:, feat_n]))
            for thres in thres_op:
                left_x, left_tmt, left_y,\
                right_x, right_tmt, right_y\
                = self.split_init(x[:, feat_n], treatment, y, threshold=thres)
                
                if self.check_min(left_tmt) and self.check_min(right_tmt):
                    ddp = self.uplift_score(left_tmt, left_y, right_tmt, right_y)
                    # select best delta delta p
                    if ddp > best_ddp:
                        best_ddp = ddp
                        split_feat = feat_n
                        split_thres = thres

        return split_feat, split_thres
    
    # generate threshold_options using threshold algorithm    
    def threshold_options(self, column_values):
        unique_values = np.unique(column_values)
        if len(unique_values) > 10:
            percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])
        return np.unique(percentiles)
    
    # calculate average treatment effect
    def average_treatment_effect(self, treatment: np.ndarray, y: np.ndarray) -> float:
        ate = y[treatment == 1].mean() - y[treatment == 0].mean()
        return ate
    
    def uplift_score(self, left_tmt, left_y, right_tmt, right_y):
        M_left = self.average_treatment_effect(left_tmt, left_y)
        M_right = self.average_treatment_effect(right_tmt, right_y)
        uplift = abs(M_left - M_right)
        return uplift
    
    # recursively run the build_tree
    def build_tree(self, x, treatment, y, depth: int = 0) -> Node:
        node = Node(n_samples=x.shape[0], ddp=0, ate=self.average_treatment_effect(treatment,y))
        if depth < self.max_depth:
            # get the best split
            feat_n, thres = self.best_split(x, treatment, y)
            
            if feat_n is not None:
                # left tree variables
                indices_left = x[:, feat_n] <= thres
                left_x = x[indices_left]
                left_treatment = treatment[indices_left]
                left_y = y[indices_left]
                
                # right tree vaiables
                right_x = x[~indices_left] 
                right_treatment = treatment[~indices_left] 
                right_y = y[~indices_left]
                
                node.split_feature = feat_n
                node.split_threshold = thres
                node.left = self.build_tree(left_x, left_treatment, left_y, depth + 1)
                node.right = self.build_tree(right_x, right_treatment, right_y, depth + 1)
        return node
    
    def fit(
       self,
       x: np.ndarray, # (n * k) array with features
       treatment: np.ndarray,  # (n) array with treatment flag
       y: np.ndarray  # (n) array with the target
    ) -> None:
       
        # fit the model
        self.n_features = len(x[0,:])
        self.tree = self.build_tree(x, treatment, y)
    
    def recurse(self, row: np.ndarray) -> float:
        node = self.tree
        while node.left:
            if row[node.split_feature] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.ate

    def predict(self, x: np.ndarray) -> Iterable[float]:
        return np.array([self.recurse(row) for row in x])

    def check_min(self, treatment: np.ndarray) -> bool:
        if (len(treatment) >= self.min_samples_leaf and
            len(treatment[treatment == 1]) >= self.min_samples_leaf_treated and
            len(treatment[treatment == 0]) >= self.min_samples_leaf_control):
            return True
        return False
    
    def split_init(self, x, treatment, y, threshold):
        left_x = x[x <= threshold]
        left_treatment = treatment[x <= threshold]
        left_y = y[x <= threshold]
        
        right_x = x[x > threshold]
        right_treatment = treatment[x > threshold]
        right_y = y[x > threshold]

        return left_x, left_treatment, left_y, right_x, right_treatment, right_y


if __name__ == '__main__':
    treatment = np.load('example_treatment.npy')
    x = np.load('example_X.npy')
    y = np.load('example_y.npy')
    ytest = np.load('example_preds.npy')    
    
    model = UpliftTreeRegressor()
    model.fit(x, treatment, y)
    ypred = model.predict(x)
    
    mse = np.square(np.subtract(ypred, ytest)).mean()
    print("mse loss ", mse)
    