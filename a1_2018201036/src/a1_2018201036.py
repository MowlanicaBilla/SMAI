#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15,7]

from sklearn import tree
from scipy.stats.mstats import mquantiles
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


# In[2]:


train_df = pd.read_csv('../input/train.csv')


# In[3]:


train_df.describe()


# In[4]:


train_df.info()


# In[5]:


mpsales = {"sales":0,
 "accounting":1,
 "technical":2,
 "management":3,
 "IT":4,
 "product_mng":5,
 "marketing":6,
 "RandD":7,
 "support":8,
 "hr":9}
train_df['sales'] = train_df['sales'].map(mpsales)


# In[6]:


mpsalary = {'low':0,
        'medium':1,
        'high':2}
train_df['salary'] = train_df['salary'].map(mpsalary)


# In[7]:


train_df.info()


# In[8]:


train_df.describe()


# ## Part-1: Train decision tree only on categorical data. Report precision, recall, f1 score and accuracy

# 1. In this problem I've considered - number_project, time_spend_company, Work_accident, promotion_last_5years, sales, salary - as categorical features and used it train my decision tree model.
# 
# 2. I've splitted the training data of 11238 size into 80:20 ratio of training and validation set
# 
# 3. I've used ID3 algorithm to build the decision tree, ie, for each categorical data with k values, I do a k way split for the node.
# 
# 4. The default algorithm to calculate impurity loss of features is, cross entropy method - Lcross(R) = Lcross(ˆp) = −pˆlog ˆp − (1 − pˆ) log (1 − pˆ), apart from that even gini index and misclassification can be used by passing it as parameter
# 
# 5. Each node calculates the impurity for the available features and the nodes is splitted based on the feature with smallest impurity
# 
# 6. There is even a max_depth parameter that can be used to limit the depth of the tree

# In[9]:


class DecisionTreeLeaf(object):
    def __init__(self,value):
        self.value = value
    def predict(self, x):
        return self.value
    def print_tree(self, indent):
        spaces = ' '*indent
        print('{0} ==> {1}'.format(spaces, self.value))


# In[10]:


class DecisionTreeBranch(object):
    def __init__(self, feature_name, subtrees, default_value):
        self.feature_name = feature_name
        self.subtrees = subtrees
        self.default_value = default_value
        
    def predict(self, x):
        subtree = self.subtrees.get(x[self.feature_name])
        if subtree:
            return subtree.predict(x)
        else:
            return self.default_value
    def print_tree(self, indent):
        spaces = ' '*indent
        print('{0}{1}:'.format(spaces, self.feature_name))
        print('{0} <default> ==> {1}'.format(spaces, self.default_value))
        for v in sorted(self.subtrees):
            print('{0} {1} ->'.format(spaces,v))
            self.subtrees[v].print_tree(indent+4)


# In[11]:


def train_decision_tree(XY, available_feature, max_depth, loss_func):
    distribution = Counter(y for _,y in XY)
    majority_value = distribution.most_common(1)[0][0]
    #print(distribution)
    if len(distribution) == 1:
        return DecisionTreeLeaf(majority_value)
    if not available_feature:
        return DecisionTreeLeaf(majority_value)
    if max_depth==0:
        return DecisionTreeLeaf(majority_value)
    selected_feature = min(available_feature, key=lambda f:impurity_scorer(f,XY, loss_func))
    #print(selected_feature)
    next_available_feature = set(available_feature) - set([selected_feature])
    #print(next_available_feature)
    XY_split = split_by_feature(selected_feature, XY)
    subtrees = {}
    for value, XY_subset in XY_split.items():
        subtrees[value] = train_decision_tree(XY_subset, next_available_feature, max_depth-1, loss_func)
    return DecisionTreeBranch(selected_feature, subtrees, majority_value)

def split_by_feature(feature_name, XY):
    XY_split = defaultdict(list)
    for x, y in XY:
        #print(x)
        XY_split[x.get(feature_name)].append((x,y))
    return XY_split

def impurity_scorer(feature_name, XY, loss_func):
    lst = list(set([x.get(feature_name) for x,y in XY]))
    #print(lst)
    q = [0]*len(lst)
    qpos = [0]*len(lst)
    for x,y in XY:
        for i in range(len(lst)):
            if x.get(feature_name) == lst[i]:
                q[i]+=1
                if y == 1:
                    qpos[i]+=1
    majority_sum = 0.0
    err = 0
    for i in range(len(q)):
        #majority_sum += 2*qpos[i]*(q[i]-qpos[i])
        #print('{0} {1}'.format(q[i],qpos[i]))
        if loss_func == 'entropy':
            if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                err = -((qpos[i]/q[i])*np.log(qpos[i]/q[i]) + ((q[i]-qpos[i])/q[i])*np.log((q[i]-qpos[i])/q[i]))
        elif loss_func == 'gini':
            if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                err = (qpos[i]/q[i])**2 + ((q[i]-qpos[i])/q[i])**2
            err = 1-err
        elif loss_func == 'misclassification':
            if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                err = 1 - max(qpos[i]/q[i],(q[i]-qpos[i])/q[i])
        majority_sum += (q[i]/len(XY)) *err
    #print('{0} {1}'.format(majority_sum, feature_name))
    return majority_sum


# In[12]:


class DecisionTree(object):
    def __init__(self, max_depth=997, loss_func='entropy'):
        self.max_depth = max_depth
        self.loss_func = loss_func
    
    def fit(self, X, Y):
        #XY = list(zip(train_df['sales'],train_df['salary'],train_df['promotion_last_5years'],train_df['Work_accident'],Y))
        XY = list(zip(X, Y))
        #print(XY)
        #print(Y)
        available_features = set(f for x in X for f in x)
        #print(available_features)
        self.root = train_decision_tree(XY, available_features,self.max_depth, self.loss_func)
        
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    def predict_one(self, x):
        return self.root.predict(x)
    def print_tree(self):
        return self.root.print_tree(0)


# In[13]:


#np.random.seed(1)
probs = np.random.rand(len(train_df))
training_mask = probs <= 0.8
validation_mask = probs > 0.8
X_train =  train_df[training_mask]
X_val = train_df[validation_mask]
X_train_cat, Y_train_cat = X_train.drop(['left', 'satisfaction_level','average_montly_hours',
                        'last_evaluation'], axis = 1), X_train['left']
X_val_cat, Y_val_cat = X_val.drop(['left', 'satisfaction_level','average_montly_hours',
                        'last_evaluation'], axis=1), X_val['left']


# In[14]:


X_train_catmp = X_train_cat.to_dict('records')
X_val_catmp = X_val_cat.to_dict('records')


# In[15]:


X_train_cat.describe()


# In[16]:


X_val_cat.describe()


# In[29]:


clfcat = DecisionTree(max_depth=9)
clfcat.fit(X_train_catmp, Y_train_cat)


# In[30]:


#clfcat.print_tree()


# In[31]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train_cat, Y_train_cat)


# In[32]:


g1 = clfcat.predict(X_val_catmp)
g2 = clf.predict(X_val_cat)
a1 = accuracy_score(Y_val_cat, g1)
a2 = accuracy_score(Y_val_cat, g2)

r1 = precision_recall_fscore_support(Y_val_cat, g1, average='binary')
r2 = precision_recall_fscore_support(Y_val_cat, g2, average='binary')

print('                    My Classifier          Scikit Learn')
print('Accuracy_score: {0}     {1}'.format(a1,a2))
print('Precision:      {0}     {1}'.format(r1[0],r2[0]))
print('Recall:         {0}     {1}'.format(r1[1],r2[1]))
print('f1 Score:       {0}     {1}'.format(r1[2],r2[2]))

print('                              My categorical classifier')
print(classification_report(Y_val_cat, g1, target_names=['No','Yes']))
print('                                scikit learn classifier')
print(classification_report(Y_val_cat, g2, target_names=['No','Yes']))


# ###### Report: 
# 1. The performance of my model is comparable to that of scikit learn decision tree classifier. 
# 2. Best perfomance on max_depth of 9 and loss_function as 'gini'. 
# 3. initialise the decision tree using - clf = DecisionTree(max_depth=x, loss_func='entropy')
# 4. Fit the decision to test data using - clf.fit(X_train, Y_train)
# 5. Predict output using - clf.predict(X_test)

# ## Part-2: Train the decision tree with categorical and numerical features. Report precision, recall, f1 score and accuracy

# 1. In this problem I've considered - satisfaction_level, number_project, average_montly_hours, time_spend_company,
#    last_evaluation - as training features and used it train my decision tree model.
# 
# 2. I've splitted the training data of 11238 size into 80:20 ratio of training and validation set
# 
# 3. I've used ID3 algorithm to build the decision tree, ie, for each categorical data with k values, I do a k way split for the node. For numerical data I find the best split point among the given data and split the tree into two parts - one for less than split_pos and the other for greater than equal.
# 
# 4. The default algorithm to calculate impurity loss of features is, cross entropy method - Lcross(R) = Lcross(ˆp) = −pˆlog ˆp − (1 − pˆ) log (1 − pˆ), apart from that even gini index and misclassification can be used by passing it as parameter
# 
# 5. Each node calculates the impurity for the available features and the nodes is splitted based on the feature with smallest impurity
# 
# 6. There is even a max_depth parameter that can be used to limit the depth of the tree

# In[36]:


nodecount = 0


# In[37]:


class DecisionTreeBranchCatNum(object):
    def __init__(self, feature_name, subtrees, default_value, split_pos):
        self.feature_name = feature_name
        self.subtrees = subtrees
        self.default_value = default_value
        self.split_pos = split_pos
        
    def predict(self, x):
        if any(self.feature_name in n_f for n_f in num_features):
            subtree = None
            if x[self.feature_name]<self.split_pos:
                subtree = self.subtrees.get(0)
            else:
                subtree = self.subtrees.get(1)
        else:
            subtree = self.subtrees.get(x[self.feature_name])
        #print(x[self.feature_name])
        if subtree:
            return subtree.predict(x)
        else:
            return self.default_value
    def print_tree(self, indent):
        spaces = ' '*indent
        print('{0}{1}:'.format(spaces, self.feature_name))
        print('{0} <default> ==> {1}'.format(spaces, self.default_value))
        for v in sorted(self.subtrees):
            print('{0} {1} ->'.format(spaces,v))
            self.subtrees[v].print_tree(indent+4)


# In[38]:


def train_decision_tree_num(XY, available_feature, max_depth, loss_func):
    distribution = Counter(y for _,y in XY)
    majority_value = distribution.most_common(1)[0][0]
    #print(distribution)
    global nodecount
    if max_depth == 0:
        nodecount+=1
        return DecisionTreeLeaf(majority_value)
    if len(distribution) == 1:
        nodecount+=1
        return DecisionTreeLeaf(majority_value)
    if not available_feature:
        nodecount+=1
        return DecisionTreeLeaf(majority_value)
    split_pos = 0
    selected_feature = min(available_feature, key=lambda f:impurity_scorer(f,XY, loss_func))
    next_available_feature = None
    if any(selected_feature in n_f for n_f in num_features):
        _,split_pos = impurity_scorer_num(selected_feature,XY, loss_func)
        XY_split = split_by_feature_num(selected_feature, XY,split_pos)
        next_available_feature = available_feature
        nodecount+=2
        #print(split_pos)
    else:
        next_available_feature = set(available_feature)  - set([selected_feature])
        #print(next_available_feature)
        XY_split = split_by_feature_cat(selected_feature, XY)
        nodecount+=len(XY_split)
    #print(selected_feature)
    #print(split_pos)
    subtrees = {}
    for value, XY_subset in XY_split.items():
        subtrees[value] = train_decision_tree_num(XY_subset, next_available_feature, max_depth-1, loss_func)
    return DecisionTreeBranchCatNum(selected_feature, subtrees, majority_value, split_pos)

def split_by_feature_num(feature_name, XY,split_pos):
    XY_split = defaultdict(list)
    for x, y in XY:
        #print(x)
        if x.get(feature_name) < split_pos:
            XY_split[0].append((x,y))
        else:
            XY_split[1].append((x,y))
    return XY_split
def split_by_feature_cat(feature_name, XY):
    XY_split = defaultdict(list)
    for x, y in XY:
        #print(x)
        XY_split[x.get(feature_name)].append((x,y))
    return XY_split

def impurity_scorer_cat(feature_name, XY, loss_func):
    lst = list(set([x.get(feature_name) for x,y in XY]))
    #print(lst)
    q = [0]*len(lst)
    qpos = [0]*len(lst)
    for x,y in XY:
        for i in range(len(lst)):
            if x.get(feature_name) == lst[i]:
                q[i]+=1
                if y == 1:
                    qpos[i]+=1
    majority_sum = 0.0
    err = 0
    for i in range(len(q)):
        #majority_sum += 2*qpos[i]*(q[i]-qpos[i])
        #print('{0} {1}'.format(q[i],qpos[i]))
        if loss_func == 'entropy':
            if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                err = -((qpos[i]/q[i])*np.log(qpos[i]/q[i]) + ((q[i]-qpos[i])/q[i])*np.log((q[i]-qpos[i])/q[i]))
        elif loss_func == 'gini':
            if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                err = (qpos[i]/q[i])**2 + ((q[i]-qpos[i])/q[i])**2
            err = 1-err
        elif loss_func == 'misclassification':
            if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                err = 1 - max(qpos[i]/q[i],(q[i]-qpos[i])/q[i])
        majority_sum += (q[i]/len(XY)) *err
    #print('{0} {1}'.format(majority_sum, feature_name))
    return majority_sum, 0

def impurity_scorer_num(feature_name, XY, loss_func):
    lst = sorted(list(set([x.get(feature_name) for x,y in XY])))
    lst.extend(mquantiles(lst).tolist())
    #print(feature_name)
    #print(lst)
    split_pos = lst[0]
    mx = 1e19
    for k in range(len(lst)):
        q = [0]*2
        qpos = [0]*2
        for x,y in XY:
            if x.get(feature_name) < lst[k]:
                q[0]+=1
                if y == 1:
                    qpos[0]+=1
            else:
                q[1]+=1
                if y==1:
                    qpos[1] += 1
                        
        majority_sum = 0.0
        err=0
        for i in range(len(q)):
            #majority_sum += 2*qpos[i]*(q[i]-qpos[i])
            #print('{0} {1}'.format(q[i],qpos[i]))
            if loss_func == 'entropy':
                if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                    err = -((qpos[i]/q[i])*np.log(qpos[i]/q[i]) + ((q[i]-qpos[i])/q[i])*np.log((q[i]-qpos[i])/q[i]))
            elif loss_func == 'gini':
                if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                    err = (qpos[i]/q[i])**2 + ((q[i]-qpos[i])/q[i])**2
                err = 1-err
            elif loss_func == 'misclassification':
                if (q[i]>0 and qpos[i]>0) and qpos[i]<q[i]:
                    err = 1 - max(qpos[i]/q[i],(q[i]-qpos[i])/q[i])
            majority_sum += (q[i]/len(XY)) *err 
        if mx>=majority_sum:
            mx = majority_sum
            split_pos = lst[k]
    #print('{0} {1} {2}'.format(feature_name, mx, lst[k]))
    return (mx,split_pos)
def impurity_scorer(feature_name, XY, loss_func):
    #print(feature_name)
    
    if any(feature_name in n_f for n_f in num_features):
        #print("hi")
        #print(feature_name)
        return impurity_scorer_num(feature_name, XY, loss_func)[0]
    else:
        #print("hi")
        #print(feature_name)
        return impurity_scorer_cat(feature_name, XY, loss_func)[0]


# In[39]:


class DecisionTreeCatNum(object):
    def __init__(self, max_depth=997, loss_func='entropy'):
        self.max_depth = max_depth
        self.loss_func = loss_func
    
    def fit(self, X, Y):
        #XY = list(zip(train_df['sales'],train_df['salary'],train_df['promotion_last_5years'],train_df['Work_accident'],Y))
        XY = list(zip(X, Y))
        #print(XY)
        #print(Y)
        available_features = set(f for x in X for f in x)
        #print(available_features)
        self.root = train_decision_tree_num(XY, available_features, self.max_depth, self.loss_func)
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    def predict_one(self, x):
        return self.root.predict(x)
    def print_tree(self):
        return self.root.print_tree(0)


# In[48]:


num_features = ['satisfaction_level','number_project','average_montly_hours','time_spend_company',
                        'last_evaluation']


# In[49]:


X_catnum = train_df.drop(['left'], axis=1)


# In[50]:


#np.random.seed(40)
probs = np.random.rand(len(train_df))
training_mask = probs <= 0.8
validation_mask = probs > 0.8
train_catnum =  train_df[training_mask]
val_catnum = train_df[validation_mask]
X_train_catnum, Y_train_catnum = train_catnum.drop('left', axis = 1), train_catnum['left']
X_val_catnum, Y_val_catnum = val_catnum.drop('left', axis=1), val_catnum['left']


# In[51]:


X_train_catnummp = X_train_catnum.to_dict('records')
X_val_catnummp = X_val_catnum.to_dict('records')


# In[52]:


clfnum = DecisionTreeCatNum(max_depth=9, loss_func='entropy')
clfnum.fit(X_train_catnummp, Y_train_catnum)


# In[53]:


#clfnum.print_tree()


# In[54]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train_catnum, Y_train_catnum)


# In[55]:


g1 = clfnum.predict(X_val_catnummp)
g2 = clf.predict(X_val_catnum)
a1 = accuracy_score(Y_val_catnum, g1)
a2 = accuracy_score(Y_val_catnum, g2)

r1 = precision_recall_fscore_support(Y_val_catnum, g1, average='binary')
r2 = precision_recall_fscore_support(Y_val_catnum, g2, average='binary')

print('                    My Classifier          Scikit Learn')
print('Accuracy_score: {0}     {1}'.format(a1,a2))
print('Precision:      {0}     {1}'.format(r1[0],r2[0]))
print('Recall:         {0}     {1}'.format(r1[1],r2[1]))
print('f1 Score:       {0}     {1}'.format(r1[2],r2[2]))

print('                              My categorical classifier')
print(classification_report(Y_val_catnum, g1, target_names=['No','Yes']))
print('                                scikit learn classifier')
print(classification_report(Y_val_catnum, g2, target_names=['No','Yes']))


# ###### Report: 
# 1. The performance of my model is comparable to that of scikit learn decision tree classifier. 
# 2. Best perfomance on max_depth of 9 and loss_function as 'entropy'. 
# 3. initialise the decision tree using - clf = DecisionTreeCatNum(max_depth=x, loss_func='entropy')
# 4. Fit the decision to test data using - clf.fit(X_train, Y_train)
# 5. Predict output using - clf.predict(X_test)

# ## Part-3: Contrast the effectiveness of Misclassification rate, Gini, Entropy as impurity measures in terms of precision, recall and accuracy

# ###### I've passed different loss function as parameter to the decision tree to calculate the accuracy, precision, recall and f1 score for the model and I've represented their comparison

# In[59]:


clf1 = DecisionTreeCatNum(max_depth=10, loss_func = 'entropy')
clf1.fit(X_train_catnummp, Y_train_catnum)


# In[60]:


clf2 = DecisionTreeCatNum(max_depth=10, loss_func = 'gini')
clf2.fit(X_train_catnummp, Y_train_catnum)


# In[61]:


clf3 = DecisionTreeCatNum(max_depth=10, loss_func = 'misclassification')
clf3.fit(X_train_catnummp, Y_train_catnum)


# In[62]:


g1 = clf1.predict(X_val_catnummp)
g2 = clf2.predict(X_val_catnummp)
g3 = clf3.predict(X_val_catnummp)
a1 = accuracy_score(Y_val_catnum, g1)
a2 = accuracy_score(Y_val_catnum, g2)
a3 = accuracy_score(Y_val_catnum, g3)

r1 = precision_recall_fscore_support(Y_val_catnum, g1, average='binary')
r2 = precision_recall_fscore_support(Y_val_catnum, g2, average='binary')
r3 = precision_recall_fscore_support(Y_val_catnum, g3, average='binary')

print('                    Entropy                 gini                misclassiction')
print('Accuracy_score: {0}     {1}     {2}'.format(a1,a2,a3))
print('Precision:      {0}     {1}     {2}'.format(r1[0],r2[0],r3[0]))
print('Recall:         {0}     {1}     {2}'.format(r1[1],r2[1],r3[1]))
print('f1 Score:       {0}     {1}     {2}'.format(r1[2],r2[2],r3[2]))

print('                              Entropy')
print(classification_report(Y_val_catnum, g1, target_names=['No','Yes']))
print('                                Gini')
print(classification_report(Y_val_catnum, g2, target_names=['No','Yes']))
print('                           misclassification')
print(classification_report(Y_val_catnum, g3, target_names=['No','Yes']))


# ###### Report: 
# It can be seen that gini index performs best when used as loss function but the accuracy for entropy loss function is comparable to gini index and so it can be used as an alternative to gini index cost function. Misclassification rate isn't a good parameter to calculate accuracy because it doesn't perform good with respect to entropy or gini index

# ## Part-4: Visualise training data on a 2-dimensional plot taking one feature (attribute) on one axis and other feature on another axis. Take two suitable features to visualise decision tree boundary (Hint: use scatter plot with differentcolors for each label)

# ###### I've selected - number_project, satisfaction_level -  as best features that can represent the decision boundary well for the training data because these has least impurity
# 
# ###### Apart from that, I've represented few more graphs with different features where the decision boundary was visible

# In[36]:


import matplotlib.pyplot as plt

plt1 = train_df[['number_project','satisfaction_level','left']]
plt2 = plt1[plt1['left']==1]
plt3 = plt1[plt1['left']==0]

plt.scatter(plt3['number_project'], plt3['satisfaction_level'], label="no", alpha=0.3)
plt.scatter(plt2['number_project'], plt2['satisfaction_level'], label="yes", alpha = 0.3)
plt.xlabel('number_project')
plt.ylabel('satisfaction_level')
plt.legend()


# In[37]:


import matplotlib.pyplot as plt

plt1 = train_df[['last_evaluation','average_montly_hours','left']]
plt2 = plt1[plt1['left']==1]
plt3 = plt1[plt1['left']==0]

plt.scatter(plt3['last_evaluation'], plt3['average_montly_hours'], label="no", alpha=0.3)
plt.scatter(plt2['last_evaluation'], plt2['average_montly_hours'], label="yes", alpha = 0.3)
plt.xlabel('last_evaluation')
plt.ylabel('average_montly_hours')
plt.legend()


# In[38]:


import matplotlib.pyplot as plt

plt1 = train_df[['last_evaluation','satisfaction_level','left']]
plt2 = plt1[plt1['left']==1]
plt3 = plt1[plt1['left']==0]

plt.scatter(plt3['last_evaluation'], plt3['satisfaction_level'], label="no", alpha=0.3)
plt.scatter(plt2['last_evaluation'], plt2['satisfaction_level'], label="yes", alpha = 0.3)
plt.xlabel('last_evaluation')
plt.ylabel('satisfaction_level')
plt.legend()


# In[39]:


import matplotlib.pyplot as plt

plt1 = train_df[['average_montly_hours','satisfaction_level','left']]
plt2 = plt1[plt1['left']==1]
plt3 = plt1[plt1['left']==0]

plt.scatter(plt3['average_montly_hours'], plt3['satisfaction_level'], label="no", alpha=0.3)
plt.scatter(plt2['average_montly_hours'], plt2['satisfaction_level'], label="yes", alpha = 0.3)
plt.xlabel('average_montly_hours')
plt.ylabel('satisfaction_level')
plt.legend()


# In[40]:


import matplotlib.pyplot as plt

plt1 = train_df[['sales','satisfaction_level','left']]
plt2 = plt1[plt1['left']==1]
plt3 = plt1[plt1['left']==0]

plt.scatter(plt3['sales'], plt3['satisfaction_level'], label="no", alpha=0.3)
plt.scatter(plt2['sales'], plt2['satisfaction_level'], label="yes", alpha = 0.3)
plt.xlabel('sales')
plt.ylabel('satisfaction_level')
plt.legend()


# ## Part-5: Plot a graph of training and validation error with respect to depth of your decision tree. Also plot the training and validation error with respect to number of nodes in the decision tree

# ###### I've represented the graph for training and validation error with respect to max depth and node count, for each loss function, by taking the error for each depth and node count at each depth and plotting them as graph 

# In[41]:


lst_training_error_dep = []
lst_validation_error_dep = []
lst_training_error_node = []
lst_validation_error_node = []
for i in range(1,31):
    clf1 = DecisionTreeCatNum(max_depth=i, loss_func = 'entropy')
    nodecount = 0
    #print(nodecount)
    clf1.fit(X_train_catnummp, Y_train_catnum)
    #print(nodecount)
    gt = clf1.predict(X_train_catnummp)
    gv = clf1.predict(X_val_catnummp)
    at = accuracy_score(Y_train_catnum, gt)
    av = accuracy_score(Y_val_catnum, gv)
    lst_training_error_dep.append((i, 1-at))
    lst_validation_error_dep.append((i,1-av))
    lst_training_error_node.append((nodecount, 1-at))
    lst_validation_error_node.append((nodecount,1-av))
    print("Itration: "+str(i))


# In[42]:


X_tr = []
Y_tr = []
#print(len(lst_training_error_dep))
for i in range(len(lst_training_error_dep)):
    X_tr.append(lst_training_error_dep[i][0])
    Y_tr.append(lst_training_error_dep[i][1])
X_vl = []
Y_vl = []
#print(len(lst_training_error_dep))
for i in range(len(lst_validation_error_dep)):
    X_vl.append(lst_validation_error_dep[i][0])
    Y_vl.append(lst_validation_error_dep[i][1])

plt.plot(X_tr, Y_tr, marker='o', label='training error')
plt.plot(X_vl, Y_vl, marker='o', label='validation error')
plt.xlabel('depth')
plt.ylabel('error')
plt.legend()


# In[43]:


X_tr = []
Y_tr = []
#print(len(lst_training_error_dep))
for i in range(len(lst_training_error_node)):
    X_tr.append(lst_training_error_node[i][0])
    Y_tr.append(lst_training_error_node[i][1])
X_vl = []
Y_vl = []
#print(len(lst_training_error_dep))
for i in range(len(lst_validation_error_node)):
    X_vl.append(lst_validation_error_node[i][0])
    Y_vl.append(lst_validation_error_node[i][1])
    
plt.plot(X_tr, Y_tr, marker='o', label='training error')
plt.plot(X_vl, Y_vl, marker='o', label='validation error')
plt.xlabel('node')
plt.ylabel('error')
plt.legend()


# In[44]:


lst_training_error_dep = []
lst_validation_error_dep = []
lst_training_error_node = []
lst_validation_error_node = []
for i in range(1,31):
    clf1 = DecisionTreeCatNum(max_depth=i, loss_func = 'gini')
    nodecount = 0
    #print(nodecount)
    clf1.fit(X_train_catnummp, Y_train_catnum)
    #print(nodecount)
    gt = clf1.predict(X_train_catnummp)
    gv = clf1.predict(X_val_catnummp)
    at = accuracy_score(Y_train_catnum, gt)
    av = accuracy_score(Y_val_catnum, gv)
    lst_training_error_dep.append((i, 1-at))
    lst_validation_error_dep.append((i,1-av))
    lst_training_error_node.append((nodecount, 1-at))
    lst_validation_error_node.append((nodecount,1-av))
    print("Itration: "+str(i))


# In[45]:


X_tr = []
Y_tr = []
#print(len(lst_training_error_dep))
for i in range(len(lst_training_error_dep)):
    X_tr.append(lst_training_error_dep[i][0])
    Y_tr.append(lst_training_error_dep[i][1])
X_vl = []
Y_vl = []
#print(len(lst_training_error_dep))
for i in range(len(lst_validation_error_dep)):
    X_vl.append(lst_validation_error_dep[i][0])
    Y_vl.append(lst_validation_error_dep[i][1])
    
plt.plot(X_tr, Y_tr, marker='o', label='training error')
plt.plot(X_vl, Y_vl, marker='o', label='validation error')
plt.xlabel('depth')
plt.ylabel('error')
plt.legend()


# In[46]:


X_tr = []
Y_tr = []
#print(len(lst_training_error_dep))
for i in range(len(lst_training_error_node)):
    X_tr.append(lst_training_error_node[i][0])
    Y_tr.append(lst_training_error_node[i][1])
X_vl = []
Y_vl = []
#print(len(lst_training_error_dep))
for i in range(len(lst_validation_error_node)):
    X_vl.append(lst_validation_error_node[i][0])
    Y_vl.append(lst_validation_error_node[i][1])
    
plt.plot(X_tr, Y_tr, marker='o', label='training error')
plt.plot(X_vl, Y_vl, marker='o', label='validation error')
plt.xlabel('node')
plt.ylabel('error')
plt.legend()


# In[47]:


lst_training_error_dep = []
lst_validation_error_dep = []
lst_training_error_node = []
lst_validation_error_node = []
for i in range(1,31):
    clf1 = DecisionTreeCatNum(max_depth=i, loss_func = 'misclassification')
    nodecount = 0
    #print(nodecount)
    clf1.fit(X_train_catnummp, Y_train_catnum)
    #print(nodecount)
    gt = clf1.predict(X_train_catnummp)
    gv = clf1.predict(X_val_catnummp)
    at = accuracy_score(Y_train_catnum, gt)
    av = accuracy_score(Y_val_catnum, gv)
    lst_training_error_dep.append((i, 1-at))
    lst_validation_error_dep.append((i,1-av))
    lst_training_error_node.append((nodecount, 1-at))
    lst_validation_error_node.append((nodecount,1-av))
    print("Itration: "+str(i))


# In[48]:


X_tr = []
Y_tr = []
#print(len(lst_training_error_dep))
for i in range(len(lst_training_error_dep)):
    X_tr.append(lst_training_error_dep[i][0])
    Y_tr.append(lst_training_error_dep[i][1])
X_vl = []
Y_vl = []
#print(len(lst_training_error_dep))
for i in range(len(lst_validation_error_dep)):
    X_vl.append(lst_validation_error_dep[i][0])
    Y_vl.append(lst_validation_error_dep[i][1])
    
plt.plot(X_tr, Y_tr, marker='o', label='training error')
plt.plot(X_vl, Y_vl, marker='o', label='validation error')
plt.xlabel('depth')
plt.ylabel('error')
plt.legend()


# In[49]:


X_tr = []
Y_tr = []
#print(len(lst_training_error_dep))
for i in range(len(lst_training_error_node)):
    X_tr.append(lst_training_error_node[i][0])
    Y_tr.append(lst_training_error_node[i][1])
X_vl = []
Y_vl = []
#print(len(lst_training_error_dep))
for i in range(len(lst_validation_error_node)):
    X_vl.append(lst_validation_error_node[i][0])
    Y_vl.append(lst_validation_error_node[i][1])
    
plt.plot(X_tr, Y_tr, marker='o', label='training error')
plt.plot(X_vl, Y_vl, marker='o', label='validation error')
plt.xlabel('node')
plt.ylabel('error')
plt.legend()


# ###### Report: 
# It can be seen that training error keeps on decreasing with increase in max depth and node count but the validation error increases after a certain threshold because the decision tree overfit to training data after a certain threshold and this affects its performance on unseen data

# ## Part-6: Explain how decision tree is suitable handle missing values(few attributes missing in test samples) in data

# #### Handling missing data
# 
# #### Strategy 1: Purification by skipping
# 
#  1. Idea 1: Skip data points where any feature contains a missing value
#     - Make sure only a few data points are skipped
#     
#  2. Idea 2: Skip an entire feature if it’s missing for many data points
#     - Make sure only a few features are skipped 
#     
# ###### Pro and Con:
# 
# ###### Pros
# 
# • Easy to understand and implement
# 
# • Can be applied to any model(decision trees, logistic regression, linear regression,…)
# 
# ###### Cons
# 
# • Removing data points and features may remove important information from data
# 
# • Unclear when it’s better to remove data points versus features
# 
# • Doesn’t help if data is missing at prediction time 
# 
# #### Strategy 2:  Purification by imputing 
# 
# 1. Impute each feature with missing values: Categorical features use mode: Most popular value (mode) of non-missing xi
#    
# 2. Numerical features use average or median: Average or median value of non-missing xi
#    
# ###### Pro and Con:
# 
# ###### Pros
# 
# • Easy to understand and implement
# 
# • Can be applied to any model (decision trees, logistic regression, linear regression,…)
# 
# • Can be used at prediction time: use same imputation rules
# 
# ###### Cons
# 
# • May result in systematic errors Example: Feature “age” missing in all banks in Washington by state law 
# #### Strategy 3: Have a default branch for each node such that this branch always gets called when there is missing data or unknown value
# 
# #### Strategy 4: Adapt learning algorithm to be robust to missing values 
# 
# ###### Feature split selection algorithm with missing value handling
# 
# • Given a subset of data M (a node in a tree)
# 
# • For each feature hi(x):
#    1. Split data points of M where hi(x) is not “unknown” according to feature hi(x)
#    2. Consider assigning data points with “unknown” value for hi(x) to each branch
#       - Compute classification error split & branch assignment of “unknown” values  
#       
# • Chose feature h*(x) & branch assignment of “unknown” with lowest classification error

# In[ ]:




