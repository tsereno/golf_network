# -*- coding: utf-8 -*-
"""import_graph_csv.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11t2WWfsPLNYVUFugb21mVluQ__ptXxbN
"""

#!pip install dgl
#!pip install BorutaShap
#!pip install umap-learn
#!pip install plotly
#!pip install lightgbm
#!pip install python-igraph
#!sudo apt install graphviz
#!pip install pygam
#!pip install factor_analyzer
#!pip install lingam
import pandas as pd
import numpy as np
import graphviz
import lingam
import dtale
import cv2
import os
import sys
import random
np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_columns = None
from BorutaShap import BorutaShap
import lightgbm as lgb
import dgl
from dgl.data import DGLDataset
import torch
torch.set_printoptions(edgeitems=10000)
from dgl.nn import GraphConv
from dgl.nn import GNNExplainer
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from umap import UMAP
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.special import softmax

pose_tubuh = ['NOSE_X',
              'NOSE_Y',
              'NOSE_Z',
              'LEFT_EYE_INNER_X', 
              'LEFT_EYE_INNER_Y', 
              'LEFT_EYE_INNER_Z', 
              'LEFT_EYE_X', 
              'LEFT_EYE_Y', 
              'LEFT_EYE_Z', 
              'LEFT_EYE_OUTER_X', 
              'LEFT_EYE_OUTER_Y', 
              'LEFT_EYE_OUTER_Z', 
              'RIGHT_EYE_INNER_X', 
              'RIGHT_EYE_INNER_Y', 
              'RIGHT_EYE_INNER_Z', 
              'RIGHT_EYE_X', 
              'RIGHT_EYE_Y', 
              'RIGHT_EYE_Z', 
              'RIGHT_EYE_OUTER_X', 
              'RIGHT_EYE_OUTER_Y', 
              'RIGHT_EYE_OUTER_Z', 
              'LEFT_EAR_X', 
              'LEFT_EAR_Y', 
              'LEFT_EAR_Z', 
              'RIGHT_EAR_X', 
              'RIGHT_EAR_Y', 
              'RIGHT_EAR_Z', 
              'MOUTH_LEFT_X', 
              'MOUTH_LEFT_Y', 
              'MOUTH_LEFT_Z', 
              'MOUTH_RIGHT_X',
              'MOUTH_RIGHT_Y',
              'MOUTH_RIGHT_Z',
              'LEFT_SHOULDER_X',
              'LEFT_SHOULDER_Y',
              'LEFT_SHOULDER_Z',
              'RIGHT_SHOULDER_X', 
              'RIGHT_SHOULDER_Y', 
              'RIGHT_SHOULDER_Z', 
              'LEFT_ELBOW_X', 
              'LEFT_ELBOW_Y', 
              'LEFT_ELBOW_Z', 
              'RIGHT_ELBOW_X', 
              'RIGHT_ELBOW_Y', 
              'RIGHT_ELBOW_Z', 
              'LEFT_WRIST_X', 
              'LEFT_WRIST_Y', 
              'LEFT_WRIST_Z', 
              'RIGHT_WRIST_X', 
              'RIGHT_WRIST_Y', 
              'RIGHT_WRIST_Z', 
              'LEFT_PINKY_X', 
              'LEFT_PINKY_Y', 
              'LEFT_PINKY_Z', 
              'RIGHT_PINKY_X', 
              'RIGHT_PINKY_Y', 
              'RIGHT_PINKY_Z', 
              'LEFT_INDEX_X', 
              'LEFT_INDEX_Y', 
              'LEFT_INDEX_Z', 
              'RIGHT_INDEX_X', 
              'RIGHT_INDEX_Y', 
              'RIGHT_INDEX_Z', 
              'LEFT_THUMB_X',
              'LEFT_THUMB_Y',
              'LEFT_THUMB_Z',
              'RIGHT_THUMB_X', 
              'RIGHT_THUMB_Y', 
              'RIGHT_THUMB_Z', 
              'LEFT_HIP_X', 
              'LEFT_HIP_Y', 
              'LEFT_HIP_Z', 
              'RIGHT_HIP_X', 
              'RIGHT_HIP_Y', 
              'RIGHT_HIP_Z', 
              'LEFT_KNEE_X', 
              'LEFT_KNEE_Y', 
              'LEFT_KNEE_Z_Z', 
              'RIGHT_KNEE_X', 
              'RIGHT_KNEE_Y', 
              'RIGHT_KNEE_Z', 
              'LEFT_ANKLE_X', 
              'LEFT_ANKLE_Y', 
              'LEFT_ANKLE_Z', 
              'RIGHT_ANKLE_X', 
              'RIGHT_ANKLE_Y', 
              'RIGHT_ANKLE_Z', 
              'LEFT_HEEL_X', 
              'LEFT_HEEL_Y', 
              'LEFT_HEEL_Z', 
              'RIGHT_HEEL_X', 
              'RIGHT_HEEL_Y', 
              'RIGHT_HEEL_Z', 
              'LEFT_FOOT_INDEX_X', 
              'LEFT_FOOT_INDEX_Y', 
              'LEFT_FOOT_INDEX_Z', 
              'RIGHT_FOOT_INDEX_X',
              'RIGHT_FOOT_INDEX_Y',
              'RIGHT_FOOT_INDEX_Z',
              'ball1',
              'ball2',
              'ball3',
              'ball4',
              'ball5',
              'flight1',
              'flight2',
              'flight3',
              'flight4',
              'flight5',
              'flight6',
              'flight7',
              'club1',
              'club2',
              'adjustment1',
              'adjustment2',
              'adjustment3',
              'adjustment4',
              'adjustment5',
              'label',
              'graph_id'
              ,'name']


"""## Load graph data from CSV"""
epochs_graph = 100
epochs_node = 100
batch_size = 1

"""#Graph Classification"""

class MyDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='custom_dataset')

    def process(self):
        self.num_classes=2
        self.graphs = []
        self.labels = []
        node_features = torch.from_numpy(nodes_data.drop(columns=['label']).to_numpy()).float()
        node_labels = torch.from_numpy(nodes_data['label'].to_numpy()).int()
        node_graphs = torch.from_numpy(nodes_data['graph_id'].to_numpy()).int()
        #edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])#self_loop=True
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        #self.graph.edata['weight'] = edge_features
        self.graph.ndata['graph_id'] = node_graphs

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties_data.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges_data.groupby('graph_id')
        nodes_group = nodes_data.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            #g = dgl.graph((src, dst))
            nodes_of_id = nodes_group.get_group(graph_id)
            node_features = torch.from_numpy(nodes_of_id.drop(columns=['label']).to_numpy()).float()
            node_labels = torch.from_numpy(nodes_of_id['label'].to_numpy()).int()
            node_graphs = torch.from_numpy(nodes_of_id['graph_id'].to_numpy()).int()
            g.ndata['feat'] = node_features
            g.ndata['label'] = node_labels
            g.ndata['graph_id'] = node_graphs
            n_nodes = num_nodes
            n_train = int(n_nodes * 0.4)
            n_val = int(n_nodes * 0.1)
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            g.ndata['train_mask'] = train_mask
            g.ndata['val_mask'] = val_mask
            g.ndata['test_mask'] = test_mask
            g = dgl.add_self_loop(g)
            self.graphs.append(g)
            self.labels.append(label)


        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    #def __getitem__(self, i):
    #    return self.graph

    #def __len__(self):
    #    return 1

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

directory = './'+sys.argv[1]+"/"
#list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(directory, x)),
#                        os.listdir(directory) ) )
'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def get_num_lines(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

# Get the list of all files in directory tree at given path
list_of_files = getListOfFiles(directory)
#random.shuffle(list_of_files)

#Train data
graphID = 0
nodes_data = pd.DataFrame()
edges_data = pd.DataFrame()
properties_data = pd.DataFrame()
temp_nodes_data = pd.DataFrame()
temp_edges_data = pd.DataFrame()
temp_properties_data = pd.DataFrame()

for entry in list_of_files:
       if entry.endswith('.csv') and "train" in entry:
        print(entry)
        subdirname = os.path.basename(os.path.dirname(entry))
        #print(subdirname)
        
        temp_data = pd.read_csv(entry)
        temp_data = temp_data[temp_data['label'] == 1]
        #temp_data = temp_data.head(1)
        temp_data = temp_data.tail(1)
        temp_data['graph_id'] = graphID
        #temp_data['label'] = int(subdirname)
        temp_data['name'] = entry
        #if temp_data.label[0] != -1:
        nodes_data = pd.concat([nodes_data, temp_data], axis="rows", ignore_index=True)
        temp_properties_data = temp_data[:1][['graph_id','name']]
        temp_properties_data['label'] = int(subdirname)
        temp_properties_data['num_nodes']=temp_data.shape[0]
        properties_data = pd.concat([properties_data, temp_properties_data], axis="rows", ignore_index=True)
        #properties_data=properties_data[:1]
        graphID=graphID+1
        temp_edges_data = temp_data[['graph_id']]
        temp_edges_data.insert(1, "src", 0)
        temp_edges_data.insert(1, "dst", 1)
        ##for index, row in temp_data.iterrows():
         ##temp_edges_data.iloc[index,1]=index
         ##temp_edges_data.iloc[index,2]=index+1
        temp_edges_data.drop(temp_edges_data.tail(1).index,inplace=True)
        edges_data = pd.concat([edges_data, temp_edges_data], axis="rows", ignore_index=True)
df = nodes_data
nodes_data = nodes_data.select_dtypes(['number'])

train_dataset = MyDataset()
g, label = train_dataset[:]
print(g, label)

#Test data
#graphID = 0
nodes_data = pd.DataFrame()
edges_data = pd.DataFrame()
#properties_data = pd.DataFrame()
temp_nodes_data = pd.DataFrame()
temp_edges_data = pd.DataFrame()
temp_properties_data = pd.DataFrame()

for entry in list_of_files:
       if entry.endswith('.csv') and "test" in entry:
        print(entry)
        subdirname = os.path.basename(os.path.dirname(entry))
        #print(subdirname)
        
        temp_data = pd.read_csv(entry)
        temp_data = temp_data[temp_data['label'] == 1]
        #temp_data = temp_data.head(1)
        temp_data = temp_data.tail(1)
        temp_data['graph_id'] = graphID
        #temp_data['label'] = int(subdirname)
        temp_data['name'] = entry
        #if temp_data.label[0] != -1:
        nodes_data = pd.concat([nodes_data, temp_data], axis="rows", ignore_index=True)
        temp_properties_data = temp_data[:1][['graph_id','name']]
        temp_properties_data['label'] = int(subdirname)
        temp_properties_data['num_nodes']=temp_data.shape[0]
        properties_data = pd.concat([properties_data, temp_properties_data], axis="rows", ignore_index=True)
        #properties_data=properties_data[:1]
        graphID=graphID+1
        temp_edges_data = temp_data[['graph_id']]
        temp_edges_data.insert(1, "src", 0)
        temp_edges_data.insert(1, "dst", 1)
        ##for index, row in temp_data.iterrows():
         ##temp_edges_data.iloc[index,1]=index
         ##temp_edges_data.iloc[index,2]=index+1
        temp_edges_data.drop(temp_edges_data.tail(1).index,inplace=True)
        edges_data = pd.concat([edges_data, temp_edges_data], axis="rows", ignore_index=True)

df = pd.concat([df, nodes_data])
df.columns = pose_tubuh
#df=df.dropna()
df.to_csv('golfswings.csv', index = False)
        
nodes_data = nodes_data.select_dtypes(['number'])


#d = dtale.show(df)
#d.open_browser()


test_dataset = MyDataset()

g, label = test_dataset[:]
print(g, label)

print(properties_data)

num_examples = len(train_dataset)
num_train = int(num_examples * 1)
num_test = len(test_dataset)

#train_sampler = SubsetRandomSampler(torch.arange(num_train))
#test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    #train_dataset, sampler=train_sampler, batch_size=1, drop_last=False)
    train_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
test_dataloader = GraphDataLoader(
    test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    ##test_dataset, sampler=test_sampler, batch_size=1, drop_last=False, shuffle=False)


#Feature_Selector_Binary_Classification = BorutaShap(model = lgb.LGBMClassifier(num_iterations=300, learning_rate=.01, verbose=1), importance_measure='shap', classification=True)
#Feature_Selector_Binary_Classification.fit(X=df.drop(columns=['label']), y=df['label'], n_trials=50, sample=True, train_or_test='test', normalize=True, verbose=True, random_state=42)
#Feature_Selector_Binary_Classification.plot(which_features='accepted', figsize=(16,12))
new_df = df
#new_df = df[df['label'] != 0]
new_df = new_df.drop(columns=[
              'graph_id',
              'name',
              'label'
              #'ball1',
              #'ball2',
              #'ball3',
              #'ball4',
              #'ball5',
              #'flight3',
              #'flight4',
              #'flight5',
              #'flight6',
              #'flight7',
              #'club1',
              #'club2'
         ])

Feature_Selector_Carry_Regression = BorutaShap(model = lgb.LGBMRegressor(num_iterations=10, learning_rate=.01, verbose=1), importance_measure='shap', classification=False)
Feature_Selector_Carry_Regression.fit(X=new_df[new_df['flight2'] > 140].drop(columns=['flight2']), y=new_df[new_df['flight2'] > 140]['flight2'], n_trials=50, sample=False, train_or_test='train', normalize=True, verbose=True, random_state=42)
Feature_Selector_Carry_Regression.plot(which_features='accepted', figsize=(16,12))
#Feature_Selector_Carry_Regression.plot(which_features='all', figsize=(16,12))
Feature_Selector_Offline_Regression = BorutaShap(model = lgb.LGBMRegressor(num_iterations=10, learning_rate=.01, verbose=1), importance_measure='shap', classification=False)
Feature_Selector_Offline_Regression.fit(X=new_df[new_df['flight1'] != 0].drop(columns=['flight1']), y=new_df[new_df['flight1'] != 0]['flight1'], n_trials=50, sample=False, train_or_test='train', normalize=True, verbose=True, random_state=42)
Feature_Selector_Offline_Regression.plot(which_features='accepted', figsize=(16,12))
#Feature_Selector_Offline_Regression.plot(which_features='all', figsize=(16,12))
Feature_Selector_Swing_Speed_Regression = BorutaShap(model = lgb.LGBMRegressor(num_iterations=10, learning_rate=.01, verbose=1), importance_measure='shap', classification=False)
Feature_Selector_Swing_Speed_Regression.fit(X=new_df[new_df['club1'] >= 60].drop(columns=['club1']), y=new_df[new_df['club1'] >= 60]['club1'], n_trials=50, sample=False, train_or_test='train', normalize=True, verbose=True, random_state=42)
Feature_Selector_Swing_Speed_Regression.plot(which_features='accepted', figsize=(16,12))
#Feature_Selector_Swing_Speed_Regression.plot(which_features='all', figsize=(16,12))

# If no model is selected default is the Random Forest
# If classification is True it is a classification problem
#clf = lgb.LGBMClassifier(num_iterations=1, learning_rate=.01, verbose=1)
#clf.fit(df.drop(columns=['label','graph_id']), df['label'])
#y_clf_pred=clf.predict(df.drop(columns=['label','graph_id']))

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot', name='GolfSwing', directory='./')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=f'{coef:.2f}')
    return d

new_df = new_df[new_df['flight2'] > 140]
new_df = new_df[new_df['flight1'] != 0]
new_df = new_df[new_df['club1'] >= 60]

'''
reg = lgb.LGBMRegressor(num_iterations=300, learning_rate=.01, verbose=0)
reg.fit(new_df.drop(columns=['flight2']), new_df['flight2'])
model = lingam.DirectLiNGAM()
model.fit(new_df)
labels = [f'{i}. {col}' for i, col in enumerate(new_df.columns)]
golfGraph = make_graph(model.adjacency_matrix_, labels)
#golfGraph.render(format='png')
# Using the causal_order_ properties, 
# we can see the causal ordering as a result of the causal discovery.
#print(model.causal_order_)
# Also, using the adjacency_matrix_ properties, 
# we can see the adjacency matrix as a result of the causal discovery.
#print(model.adjacency_matrix_)
ce = lingam.CausalEffect(model)
effects = ce.estimate_effects_on_prediction(new_df, 105, reg)
df_effects = pd.DataFrame()
df_effects['feature'] = new_df.columns
df_effects['effect_plus'] = effects[:, 0]
df_effects['effect_minus'] = effects[:, 1]
df_effects.to_csv('Carry_Feature_Causal_Importance.csv', index = False)
max_index = np.unravel_index(np.argmax(effects), effects.shape)
print("Greatest Cause: ", new_df.columns[max_index[0]])

reg = lgb.LGBMRegressor(num_iterations=300, learning_rate=.01, verbose=0)
reg.fit(new_df.drop(columns=['flight1']), new_df['flight1'])
model = lingam.DirectLiNGAM()
model.fit(new_df)
labels = [f'{i}. {col}' for i, col in enumerate(new_df.columns)]
golfGraph = make_graph(model.adjacency_matrix_, labels)
#golfGraph.render(format='png')
# Using the causal_order_ properties, 
# we can see the causal ordering as a result of the causal discovery.
#print(model.causal_order_)
# Also, using the adjacency_matrix_ properties, 
# we can see the adjacency matrix as a result of the causal discovery.
#print(model.adjacency_matrix_)
ce = lingam.CausalEffect(model)
effects = ce.estimate_effects_on_prediction(new_df, 104, reg)
df_effects = pd.DataFrame()
df_effects['feature'] = new_df.columns
df_effects['effect_plus'] = effects[:, 0]
df_effects['effect_minus'] = effects[:, 1]
df_effects.to_csv('Offline_Feature_Causal_Importance.csv', index = False)
max_index = np.unravel_index(np.argmax(effects), effects.shape)
print("Greatest Cause: ", new_df.columns[max_index[0]])

reg = lgb.LGBMRegressor(num_iterations=300, learning_rate=.01, verbose=0)
reg.fit(new_df.drop(columns=['club1']), new_df['club1'])
model = lingam.DirectLiNGAM()
model.fit(new_df)
labels = [f'{i}. {col}' for i, col in enumerate(new_df.columns)]
golfGraph = make_graph(model.adjacency_matrix_, labels)
#golfGraph.render(format='png')
# Using the causal_order_ properties, 
# we can see the causal ordering as a result of the causal discovery.
#print(model.causal_order_)
# Also, using the adjacency_matrix_ properties, 
# we can see the adjacency matrix as a result of the causal discovery.
#print(model.adjacency_matrix_)
ce = lingam.CausalEffect(model)
effects = ce.estimate_effects_on_prediction(new_df, 111, reg)
df_effects = pd.DataFrame()
df_effects['feature'] = new_df.columns
df_effects['effect_plus'] = effects[:, 0]
df_effects['effect_minus'] = effects[:, 1]
df_effects.to_csv('Speed_Feature_Causal_Importance.csv', index = False)
max_index = np.unravel_index(np.argmax(effects), effects.shape)
print("Greatest Cause: ", new_df.columns[max_index[0]])


new_df1 = new_df.drop(columns=Feature_Selector_Carry_Regression.rejected)
reg = lgb.LGBMRegressor(num_iterations=300, learning_rate=.01, verbose=0)
reg.fit(new_df1.drop(columns=['flight2','label']), new_df1['flight2'])
y_reg_pred=reg.predict(new_df1.drop(columns=['flight2','label']))
temp_df = df
temp_df['flight2_prediction']=y_reg_pred

new_df2 = new_df.drop(columns=Feature_Selector_Offline_Regression.rejected)
reg = lgb.LGBMRegressor(num_iterations=300, learning_rate=.01, verbose=0)
reg.fit(new_df2.drop(columns=['flight1','label']), new_df2['flight1'])
y_reg_pred=reg.predict(new_df2.drop(columns=['flight1','label']))
temp_df['flight1_prediction']=y_reg_pred

new_df3 = new_df.drop(columns=Feature_Selector_Swing_Speed_Regression.rejected)
reg = lgb.LGBMRegressor(num_iterations=300, learning_rate=.01, verbose=0)
reg.fit(new_df3.drop(columns=['label','club1']), new_df3['club1'])
y_reg_pred=reg.predict(new_df3.drop(columns=['label','club1']))
temp_df['club1_prediction']=y_reg_pred

temp_df['carry-offline+clubhead_speed_actual']=temp_df['flight2']-temp_df['flight1'].abs()+temp_df['club1']
temp_df['carry-offline+clubhead_speed_prediction']=temp_df['flight2_prediction']-temp_df['flight1_prediction'].abs()+temp_df['club1_prediction']
temp_df.to_csv('swing_predictions.csv', index = False)

#ax = lgb.plot_importance(clf, max_num_features=20, figsize=(15,15))
#ax = lgb.plot_importance(reg, max_num_features=20, figsize=(15,15))
#plt.show()
#plt.save
'''
'''
labels = df['label']
features = df.drop(columns=['label'])

umap_2d = UMAP(n_components=2, init='spectral', random_state=0, n_neighbors=50, n_epochs=100, densmap=False, set_op_mix_ratio=0.25, metric='euclidean')
umap_3d = UMAP(n_components=3, init='spectral', random_state=0, n_neighbors=50, n_epochs=100, densmap=False, set_op_mix_ratio=0.25, metric='euclidean')

proj_2d = umap_2d.fit_transform(features,labels)
proj_3d = umap_3d.fit_transform(features,labels)

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.label, labels={'color': 'label'}
)
fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=df.label, labels={'color': 'label'}
)
fig_3d.update_traces(marker_size=5)

fig_2d.show()
fig_3d.show()
'''



#it = iter(train_dataloader)
#batch = next(it)
#print(batch)

#batched_graph, labels = batch
#print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
#print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())

# Recover the original graph elements from the minibatch
#graphs = dgl.unbatch(batched_graph)
#print('The original graphs in the minibatch:')
#print(graphs)

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, feat, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(feat, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
   
    def forward(self, graph, feat, eweight=None):
        h = self.conv1(graph, feat)
        h = F.relu(h)
        h = self.conv2(graph, h)
        graph.ndata['h'] = h
        return dgl.mean_nodes(graph, 'h')

dataset = MyDataset()
g, label = dataset[0]
print(g, label)

model = GCN(g.ndata['feat'].shape[1], g.num_nodes(), dataset.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

temp_graph_class_pred = pd.DataFrame()
graph_class_pred = pd.DataFrame()
for epoch in range(epochs_graph):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['feat'].float())
        loss = F.cross_entropy(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
     print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['feat'].float())

    #print(batched_graph)
    
    print("Label: ", labels)
    print("Probability: ", torch.nn.functional.softmax(pred))
    print("Prediction: ", pred.argmax(1), "\n")

    graph_name = properties_data['name'].iloc[batched_graph.ndata['graph_id'][0].numpy()]
    temp_graph_class_pred['graph_label'] = labels.detach().numpy()
    temp_graph_class_pred['graph_pred'] = pd.DataFrame(pred.argmax(1).detach().numpy())
    temp_graph_class_pred['graph_probability_0'] = pd.Series(torch.nn.functional.softmax(pred).detach().numpy()[:,0])
    temp_graph_class_pred['graph_probability_1'] = pd.Series(torch.nn.functional.softmax(pred).detach().numpy()[:,1])
    temp_graph_class_pred['graph_name'] = graph_name
    #temp_graph_class_pred['club'] = graph_name[:-4]
    graph_class_pred = pd.concat([graph_class_pred, temp_graph_class_pred])

    
    #Classification
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)
    
    #regression
    #from sklearn.metrics import r2_score, mean_squared_error
    #r2 = r2_score(pred.argmax(1).float(), labels)
    #print("R2 Score:", r2)
    #mse = mean_squared_error(pred.argmax(1).float(), labels)
    #print("Mean Squared Error Score:", mse)

#classification
print("num_correct:",  num_correct)
print("num_tests:",  num_tests)
print('Test accuracy:', num_correct / num_tests)

explainer = GNNExplainer(model, num_hops=1)
dataset = MyDataset()

g, _ = dataset[0]
features = g.ndata['feat']
feat_mask, edge_mask = explainer.explain_graph(g, features)
print(g)
print(feat_mask)
print(edge_mask)

#g, _ = dataset[1]
#features = g.ndata['feat']
#feat_mask, edge_mask = explainer.explain_graph(g, features)
#print(g)
#print(feat_mask)
#print(edge_mask)

new_center, sg, feat_mask, edge_mask = explainer.explain_node(30, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(31, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(32, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(33, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(34, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(35, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(36, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(37, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(38, g, features)
print(edge_mask)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(39, g, features)
print(edge_mask)

def generate_video(graph_name):
 # Create a VideoCapture object and read from input file
 cap = cv2.VideoCapture(
     graph_name, cv2.CAP_FFMPEG)
 fps = cap.get(cv2.CAP_PROP_FPS)
 print("Frame rate: ", int(fps), "FPS")
 skip_frames = round(fps/30)-1
 print("Skipping every : ", skip_frames, "frames")


 # Check if file opened successfully
 if (cap.isOpened()== False):
     print("Error opening video file")
 frame_number=1
 counter=0
 # We need to set resolutions.
 # so, convert them from float to integer.
 frame_width = int(cap.get(3))
 frame_height = int(cap.get(4))
   
 size = (frame_width, frame_height)
   
 # Below VideoWriter object will create
 # a frame of above defined The output 
 # is stored in 'filename.avi' file.
 result = cv2.VideoWriter(graph_name+'.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
 # Read until video is completed
 while(cap.isOpened()):
  # Capture frame-by-frame
     ret, frame = cap.read()
     if ret == True:
        # Display the resulting frame
        if(frame_number % skip_frames == 0):
         #if pred.argmax(1)[counter]!=0: 
         if counter < temp_node_class_pred['node_probability_1'].shape[0]:
          if temp_node_class_pred['node_probability_1'][counter] >= .50:
           #if batched_graph.ndata['label'][counter]==1:
           result.write(frame)
           cv2.imshow('Frame', frame)
        counter+=1
         
     # Press Q on keyboard to exit
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        if cv2.waitKey(25) & 0xFF ==27:
          break
 
 # Break the loop
     else:
         break
     frame_number+=1     
 # When everything done, release
 # the video capture object
 # When everything done, release 
 # the video capture and video 
 # write objects
 cap.release()
 result.release()
 
 # Closes all the frames
 cv2.destroyAllWindows()

"""#Node Classification and Regression"""

#Classification
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

#Regression
'''
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.linear1 = torch.nn.Linear(in_feats,h_feats)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        #h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(g, h)
        h = self.linear1(h)
        return h
'''
    
# Create the model with given dimensions

#dataset = CoraGraphDataset()
#g = dataset[0]
#print(g)

dataset = MyDataset()
g, label = dataset[0]

model = GCN(g.ndata['feat'].shape[1], g.num_nodes(), dataset.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#best_val_acc = 0
#best_test_acc = 0

#features = g.ndata['feat']
#labels = g.ndata['label']
#train_mask = g.ndata['train_mask']
#val_mask = g.ndata['val_mask']
#test_mask = g.ndata['test_mask']
for epoch in range(epochs_node):
 for batched_graph, labels in train_dataloader:
    #classification
    # Forward
    logits = model(batched_graph, batched_graph.ndata['feat'])
    ##logits = model(g, features)

    # Compute prediction
    #pred = logits.argmax(1)

    # Compute loss
    # Note that you should only compute the losses of the nodes in the training set.
    #classification
    loss = F.cross_entropy(logits, batched_graph.ndata['label'].type(torch.LongTensor))
    #print(loss)
    #loss = F.cross_entropy(logits, labels)
    ##loss = F.cross_entropy(logits[train_mask], labels[train_mask])

    #regression
    #outputs = model(batched_graph, batched_graph.ndata['feat']).argmax(1)
    #loss = F.mse_loss(outputs, batched_graph.ndata['label'])

    # Compute accuracy on training/validation/test
    #train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    #val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
    #test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

    # Save the best validation accuracy and the corresponding test accuracy.
    #if best_val_acc < val_acc:
    #    best_val_acc = val_acc
    #    best_test_acc = test_acc


    # Backward
    optimizer.zero_grad()
    #loss.requires_grad = True
    loss.backward()
    optimizer.step()

 if epoch % 5 == 0:
  print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
  #if e % 5 == 0:
  #    print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
  #        e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
'''
#regression
num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    graph_name = properties_data['name'].iloc[batched_graph.ndata['graph_id'][0].numpy()]
    print(graph_name[:-4])
    pred = model(batched_graph, batched_graph.ndata['feat'].float())
    print("Label: ", batched_graph.ndata['label'])
    print("Prediction: ", pred.argmax(1), "\n")
    generate_video(graph_name[:-4])
    ##num_correct += (pred.argmax(1) == labels).sum().item()
    #num_correct += (pred.argmax(1) == batched_graph.ndata['label']).sum().item()
    ##num_tests += len(labels)
    #num_tests += len(batched_graph.ndata['label'])
    #print('Number Correct:', num_correct)
    #print('Number Tests:', num_tests)
    #print('Test accuracy:', num_correct / num_tests)

    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(pred.argmax(1).float(), batched_graph.ndata['label'].int())
    print("R2 Score:", r2)
    mse = mean_squared_error(pred.argmax(1).float(), batched_graph.ndata['label'].int())
    print("Mean Squared Error Score:", mse)
    
    #tot = ((labels - labels.mean()) ** 2).sum()
    #res = ((labels - pred) ** 2).sum()
    #r2 = 1 - res / tot
    #print("Score: ", r2)
'''
#classification
num_correct = 0
num_tests = 0
temp_node_class_pred = pd.DataFrame()
node_class_pred = pd.DataFrame()
for batched_graph, labels in test_dataloader:
    graph_name = properties_data['name'].iloc[batched_graph.ndata['graph_id'][0].numpy()]
    pred = model(batched_graph, batched_graph.ndata['feat'].float())
    print("Label: ", batched_graph.ndata['label'])
    print("Probability: ", torch.nn.functional.softmax(pred), "\n")
    print("Prediction: ", pred.argmax(1), "\n")
    temp_node_class_pred = pd.DataFrame(batched_graph.ndata['feat'].detach().numpy())
    temp_node_class_pred['node_label'] = batched_graph.ndata['label'].detach().numpy()
    temp_node_class_pred['node_pred'] = pred.argmax(1).detach().numpy()
    temp_node_class_pred['node_probability_0'] = pd.Series(torch.nn.functional.softmax(pred).detach().numpy()[:,0])
    temp_node_class_pred['node_probability_1'] = pd.Series(torch.nn.functional.softmax(pred).detach().numpy()[:,1])
    temp_node_class_pred['graph_name'] = graph_name
    node_class_pred = pd.concat([node_class_pred, temp_node_class_pred])
    #generate_video(graph_name[:-4])
    num_correct += (pred.argmax(1) == batched_graph.ndata['label']).sum().item()
    num_tests += len(batched_graph.ndata['label'])
print('Number Correct:', num_correct)
print('Number Tests:', num_tests)
print('Test accuracy:', num_correct / num_tests)

# write the predictions data to csv files
graph_class_pred.to_csv('graph_classification_predictions.csv', index = False)
node_class_pred.to_csv('node_classification_predictions.csv', index = False)
print('save complete')

#d = dtale.show(graph_class_pred)
#d.open_browser()

#d = dtale.show(node_class_pred)
#d.open_browser()
