import tensorflow as tf
import sys
import csv
from scipy import sparse
import numpy as np
from collections import defaultdict
from pprint import pprint

def convert_sparse_matrix_to_sparse_tensor(connections_sparse_matrix):
    coo = connections_sparse_matrix.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def runTests(file_name):
    source_indices, destination_indices, connections = [], [], []
    out_degree = defaultdict(int)
    count = 0
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            try:
                row_id = int(row[0])
                col_id = int(row[1])
            except:
                continue
            source_indices.append(row_id)
            destination_indices.append(col_id)
            out_degree[row_id] += 1
            count += 1
    for i in range(count):
        connections.append(1/out_degree[source_indices[i]])
    connections_sparse_matrix = sparse.coo_matrix((connections, (destination_indices, source_indices)), dtype=np.float32)
    return connections_sparse_matrix

if __name__== "__main__":
    if len(sys.argv) < 2:
        print("Invalid arguments")
        print(">python3 A2_P3_TensorFlowPR_ravula_112026804.py graph_info_data.txt")
        sys.exit(0)
    file_name = sys.argv[1]

    connections_sparse_matrix = runTests(file_name)

    coo = connections_sparse_matrix.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    connections_sparse_tensor = tf.SparseTensor(indices, coo.data, coo.shape)

    rows = max(coo.shape)
    beta = tf.constant(0.85, dtype=tf.float32, name="beta")
    beta_rem = tf.constant(0.15, dtype=tf.float32, name="beta_rem")

    V = tf.Variable(tf.convert_to_tensor(np.array([1.0/rows]*rows).reshape(rows, 1), dtype=tf.float32))

    V3 = tf.convert_to_tensor(np.array([1.0/rows]*rows).reshape(rows, 1), dtype=tf.float32)
    V5 = tf.sparse.sparse_dense_matmul(connections_sparse_tensor, V)
    V6 = tf.multiply(beta, V5)
    V7 = tf.math.add(V6, beta_rem/rows)

    threshold = 1e-12

    error = tf.reduce_sum(tf.square(tf.subtract(V7, V)))
    assignment = tf.assign(V, V7)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        i = 1
        print("The value of threshold[Squared difference between Ranks of current and previous step] is ", threshold)
        while 1 == 1:
            V1_output = sess.run(V7)
            e = sess.run(error)
            print("Iteration {}: Error {}".format(i, e))
            i += 1
            if e < threshold:
                break
            sess.run(assignment)
    ranks = np.array(V1_output).reshape(rows)
    ranks_indices = np.argsort(ranks)
    actual_ranks = ranks[ranks_indices]
    print("##########################################")
    print("The top 20 node ids along with their ranks")
    print("##########################################")
    print("##########FORMAT: [(NODE, RANK)]##########")
    pprint(list(zip(ranks_indices, actual_ranks))[0:20])
    print("#############################################")
    print("The bottom 20 node ids along with their ranks")
    print("#############################################")
    print("###########FORMAT: [(NODE, RANK)]############")
    pprint(list(zip(ranks_indices, actual_ranks))[-20:][::-1])
