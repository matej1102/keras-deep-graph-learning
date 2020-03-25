import keras.backend as K
import networkx
from keras import Sequential
from keras.layers import Dropout, Activation
from keras.optimizers import Adam

import examples.utils
from definitionsV4 import generate_networkx_graphs
from initialization import *
from keras_dgl.layers import GraphCNN

last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []
best_test_loss = 9999
# @title Run training  { form-width: "30%" }

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.



input_graphs, target_graphs  = generate_networkx_graphs(rand, batch_size_tr, num_nodes_min_max_tr, theta, True)
A=[]
Y=[]
for index in range(len(input_graphs)):
        A.append(networkx.convert_matrix.to_numpy_array(input_graphs[index]))
        Y.append(networkx.convert_matrix.to_numpy_array(target_graphs[index]))
print(A)
print(Y)

SYM_NORM = True
A_norm = examples.utils.preprocess_adj_numpy(A, SYM_NORM)
num_filters = 2
graph_conv_filters = examples.utils.np.concatenate([A_norm, examples.utils.np.matmul(A_norm, A_norm)], axis=0)
graph_conv_filters = K.constant(graph_conv_filters)

model = Sequential()
model.add(GraphCNN(16, num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Dropout(0.2))
model.add(GraphCNN(Y.shape[1], num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

