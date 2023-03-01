from sklearn.datasets import load_iris
import streamlit as st
import tensorflow as tf
import networkx as nx
from igviz.igviz import plot
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

'''
# В разработке
'''

BASE_DIR = Path('./')

def draw_image(img_obj, w: tf.Variable):
    w = np.round(w.value().numpy(), 3)
    img_obj.image((NN_SVG
                   .replace("w1", str(w[0]))
                   .replace('w2', str(w[1]))
                   ), width=300)

@tf.function
def one_layer_perceptron(x, w):
    return x*w


# NN_VIS = nx.Graph()
# NN_VIS.add_edge("X1", "Y1")
# NN_VIS.add_edge("X2", "Y1")
# NN_VIS.add_edge("X3", "Y1")
# NN_VIS.add_edge("X4", "Y1")

# st.plotly_chart(plot(NN_VIS))

st.image("./res/nn_perceptron.svg", width=500)
NN_SVG = (BASE_DIR / 'res/nn_perceptron.svg').read_text()
    
W = tf.Variable(tf.random.uniform(shape=(2,)))

raw = load_iris(as_frame=True)
labels = raw.target_names
data: pd.DataFrame = raw.frame
# data  = pd.DataFrame(PCA(2).fit_transform(data.drop(columns=['target'])), columns=['x', 'y'])
data  = pd.DataFrame(data.values[:, [0, 1, 4]], columns=['x1', 'x2', 'y'])
st.write(data)
st.plotly_chart(px.scatter(data, x='x1', y='x2', color='y'))

img_obj = st.image(NN_SVG, width=300)
draw_image(img_obj, W)

if st.button("Train"):
    data_len = data.y.shape[0]
    X = tf.convert_to_tensor(data.values[:2], shape=(None, data_len), dtype=float)
    Y = tf.convert_to_tensor(data.values[-1], shape=(data_len,), dtype=float)
    
    y_ = one_layer_perceptron(X, W)
    
    loss = tf.abs(y_ - Y).mean()
    st.write(loss)