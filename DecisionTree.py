import numpy as np
import pandas as pd
import math

def _entropy(L):
    items, counts = np.unique(L, return_counts=True)
    probs = counts / counts.sum()
    return -sum(p * np.log2(p) for p in probs)

def _info_gain(data, features, target, test):
    H_X = _entropy(data[target].values)
    feature, threshold = test.feature, test.threshold
    
    left_split = data[data[feature] > threshold]
    right_split = data[data[feature] <= threshold]

    n, nL, nR = len(data), len(left_split), len(right_split)
    
    if nL == 0 or nR == 0:
        return 0

    H_XL = _entropy(left_split[target].values)
    H_XR = _entropy(right_split[target].values)
    
    IG = H_X - (nL / n) * H_XL - (nR / n) * H_XR
    return IG

class DecisionTest:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        
    def __call__(self, x):
        return x[self.feature] > self.threshold


class DecisionTree:
    def __init__(self, data, features, target, left=None, right=None, test=None):
        self.data = data
        self.features = data[features]
        self.target = data[target]
        self.target_label = target
        self.feature_labels = features
        
        self.left = left
        self.right = right
        self.test = test
        
    def terminal(self):
        return not self.left or not self.right
    
    def predict(self, x):
        if self.terminal():
            classes, counts = np.unique(self.target, return_counts=True)
            return classes[np.argmax(counts)]
        return self.right.predict(x) if self.test(x) else self.left.predict(x)
    
    def learn(self, depth=0, max_depth=math.inf):
        max_info_gain = 0
        optimal_test = None
        
        for feature in self.feature_labels:
            observations = np.unique(self.features[feature])
            for i in range(len(observations) - 1):
                threshold = (observations[i] + observations[i+1])/2
                test = DecisionTest(feature, threshold)
                info_gain = _info_gain(self.data, self.feature_labels, self.target_label, test)
                
                if info_gain > max_info_gain:
                    optimal_test = test
                    max_info_gain = info_gain
                    
        if depth >= max_depth or not optimal_test:
            return
        
        self.test = optimal_test
        self.left = DecisionTree(self.data[self.data[optimal_test.feature] <= optimal_test.threshold], self.feature_labels, self.target_label)
        self.right = DecisionTree(self.data[self.data[optimal_test.feature] > optimal_test.threshold], self.feature_labels, self.target_label)
        
        self.left.learn(depth=depth + 1, max_depth=max_depth)
        self.right.learn(depth=depth + 1, max_depth=max_depth)
