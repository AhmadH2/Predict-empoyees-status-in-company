
"""
@author: ahmad horyzat
"""

import networkx as nx
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# G is company's email network where each node corresponds to a person at the company, and each edge indicates that at least 
# one email has been sent between two people.

# The network also contains the node attributes Department and ManagementSalary.

# Department indicates the department in the company which the person belongs to, and ManagementSalary indicates whether
# that person is receiving a management position salary [0,1].


with open('email_prediction_nodes.data', 'rb') as filehandle:
    nodes_list = pickle.load(filehandle)

with open('email_prediction_edges.data', 'rb') as filehandle:
    edges_list = pickle.load(filehandle)

    
G = nx.Graph(edges_list)
G.add_nodes_from(nodes_list)

# The future connections information has been loaded into the variable future_connections. The index is a tuple 
# indicating a pair of nodes that currently do not have a connection, and the Future Connection column indicates 
# if an edge between those two nodes will exist in the future, where a value of 1.0 indicates a future connection.
future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})


# Using network G, this function identify the people in the network with missing values for the node attribute
# ManagementSalary and predict whether or not these individuals are receiving a management position salary.
# Node degree used in the training data as employees with lots of connections are more likely to become managers
def salary_predictions():
    data = G.nodes(data=True)
    nodes_data = np.array([[node[0], node[1]['Department'], nx.degree(G, node[0]), 
                            node[1]['ManagementSalary']] for node in data])

    X_train = np.array([i for i in nodes_data if ((i[-1] == 1) or (i[-1] == 0))])
    y_train = X_train[:, -1]
    X_train = X_train[:, :-1]
    X_test = np.array([i for i in nodes_data if (np.isnan(i[-1]))])[:, :-1]

    X_test_nodes = [i[0] for i in X_test]
    clf = LogisticRegression().fit(X_train, y_train)
    predicted_mange_salary = clf.predict_proba(X_test)[:, 1]
    predicted_mange_salary = pd.Series(data=predicted_mange_salary, index=X_test_nodes)

    return predicted_mange_salary


# Using network G and future_connections, this function identify the edges in future_connections with missing values 
# and predict whether or not these edges will have a future connection.
# Shared employees between the two nodes is used in training data
def new_connections_predictions():
    nodes = future_connections.index

    future_connections['Common Neighbors'] = [len(list(nx.common_neighbors(G, i[0], i[1]))) for i in nodes]
    future_connections.reset_index(inplace=True)
    future_connections['First Node'] = future_connections['index'].map(lambda edge: edge[0])
    future_connections['Second Node'] = future_connections['index'].map(lambda edge: edge[1])

    known_connections = future_connections.dropna()
    unknown_connections = future_connections[np.isnan(future_connections['Future Connection'])]

    X_train = known_connections[['First Node', 'Second Node', 'Common Neighbors']]
    y_train = known_connections['Future Connection']
    X_test = unknown_connections[['First Node', 'Second Node', 'Common Neighbors']]

    clf = LogisticRegression().fit(X_train, y_train)
    predicted_connection = clf.predict_proba(X_test)[:, 1]
    predicted_connection = pd.Series(data=predicted_connection, index=unknown_connections['index'])

    return predicted_connection



print('probability of getting Management Salary for every employee:\n{}'.format(salary_predictions()))

print('probability of two employees to perform connections in the future:\n{}'.format(new_connections_predictions()))