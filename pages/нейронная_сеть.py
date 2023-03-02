from time import sleep
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

'''
# В разработке
'''

BASE_DIR = Path('./')

def draw_image(img_obj, w):
    w = np.round(w.numpy(), 3)
    img_obj.image((NN_SVG
                   .replace("w1", str(w[0]))
                   .replace('w2', str(w[1]))
                   ), width=300)

@tf.function
def one_layer_perceptron(x, w, B):
    return tf.tensordot(x, w, axes=1) - B



st.image("./res/nn_perceptron.svg", width=500)
NN_SVG = (BASE_DIR / 'res/nn_perceptron.svg').read_text()
    

raw = load_iris(as_frame=True)
labels = raw.target_names
data: pd.DataFrame = raw.frame
# data  = pd.DataFrame(PCA(2).fit_transform(data.drop(columns=['target'])), columns=['x', 'y'])
data = pd.DataFrame(data.values[:, [0, 1, 4]], columns=['x1', 'x2', 'y'])
data = data.assign(y=data.y.apply(lambda y: y == 0).astype(int)) 
st.write(data)
st.plotly_chart(px.scatter(data, x='x1', y='x2', color='y'))


if st.button("Train"):
    W = tf.Variable(tf.random.uniform(shape=(2,)))
    B = tf.Variable(tf.random.uniform(shape=(1,)))

    img_obj = st.image(NN_SVG, width=300)
    draw_image(img_obj, W) 


    data_len = data.y.shape[0]
    X = tf.convert_to_tensor(data.values[:, :2], dtype=float)
    Y = tf.convert_to_tensor(data.values[:, -1], dtype=float)
    dataset = tf.data
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    alpha = 0.01
    for _ in range(20):
        
        y_ = one_layer_perceptron(X, W, B)
        loss = bce(Y, y_)
        W = W - tf.reduce_mean(alpha*loss*X)
        B = B - tf.reduce_mean(alpha*loss*X)

        draw_image(img_obj, W) 
        st.write(tf.reduce_mean(loss).numpy())
        st.metric("F1", f1_score())
        sleep(0.5)