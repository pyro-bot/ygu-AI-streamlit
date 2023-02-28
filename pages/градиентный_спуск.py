import io
from time import sleep
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt



raw = load_iris(as_frame=True)
labels = raw.target_names
data: pd.DataFrame = raw.frame
data = data[data.target == 0]
# data  = pd.DataFrame(PCA(2).fit_transform(data.drop(columns=['target'])), columns=['x', 'y'])
data  = pd.DataFrame(data.values[:, :2], columns=['x', 'y'])
'Имеется датасет точет'
st.write(data)
'Отобразим его'
st.plotly_chart(px.scatter(data, x='x', y='y'))

'Для уравнения'
st.latex(r'МНК = \sum_{i=1}^{n}e^{2} = \sum_{i=1}^{n} (y_{i} - f(x_{i}))^{2} = \sum_{i=1}^{n} (y_{i} - (kx + b))^2 \rightarrow min \ \ \ \ (1)')
'Напишем функцию'

def opt_func(x: np.array, y: np.array, k: float, b: float) -> float:
    return sum((y - (k*x + b))**2)

st.code("""def opt_func(x: np.array, y: np.array, k: float, b: float) -> float:
    return sum((y - (k*x + b))**2)""")

'Далее наша задача состоит в том что бы оптимизировать `k` и `b` поэтому мы на каждом шаге оптимизации стараемся уменьшить значения полученные в хоче подстановки всех текущих переменных в формулу `(1)`'

'''Снизу находится анимация процесса оптимизация уравнения `(2)`. В ней есть 3-графика:
1. График оптимизация параметра `k`
2. График оптимизация параметра `b`
3. График входных точек для которых надо найти линию регрессии и сама линия регресии, показанныя с текущими параметрами `k` и `b`

Во время оптимизациия произойдет максимус 500 итераций алгоритма, но как правило на каждый шаг хватает ~40.  
В ходе оптимизации мы увидим красную точку которая постепенно скатывается по черной параболе.   
Тут красная точка показывает текущее значение параметра `k` или `b` (ось X) и значение МНК (ось Y).  
Черная линия это вспомогательная линия показывающая график функции `(2)` в зависимости от изменения парамтра `k` или `b`  

Основыми формалами по которам находится изменения для параметров `k` и `b` являются:
'''

st.latex(r'k(t) = k(t-1) + \frac{dMНК(x, y, k(t-1), b)}{dk} = k(t-1)+\Delta{МНК(x, y, k(t-1), b)}')
st.latex(r'b(t) = b(t-1)+ \frac{dMНК(x, y, k, b(t-1))}{dk} = b(t-1) + \Delta{МКН(x, y, k, b(t-1))}')

def get_fame(x, y, px=None, py=None, e=None, n=None, k=None, b=None, title=None, xlabel=None):
    plt.plot(plant_x, plant_y, 'k')
    if title is not None:
        plt.text(0, (np.max(y) - np.min(y))*0.8, title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel('МНК')
    if e is not None:
        plt.text(4, 35_000, f'МНК = {round(e, 3)}')
    if px is not None and py is not None:
        plt.plot([px], [py], 'r.', ms=10)
    if k is not None and b is not None:
        plt.title(f'y = {round(k, 3)}*x + {round(b, 3)}')
    if n is not None:
        plt.text(4, 40_000, f'n = {n}')
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    return buf


x = data.x
y = data.y
x_ = np.linspace(data.x.min(), data.x.max(), 10)
k = -4.0
b = 0.0
e = 0.1
alpha = 0.0001
plant_x = np.linspace(-5, 5, 50)
plant_y = [opt_func(x, y, k, 0) for k in plant_x]
plt.figure(figsize=(8, 10))
plt.subplot(3, 1, 3)
img = st.image(get_fame(plant_x, plant_y))
prev_f = opt_func(x, y, k, b)
cur_f = opt_func(x, y, k-e, b)
if st.button('Оптимизировать'):    
    
    for i in range(500):
        plt.figure(figsize=(8, 10))
        plt.subplot(3, 1, 3)
        plt.plot(x, y, 'b.')
        plt.plot(x_, [k*p + b for p in x_])
        plt.xlabel('x')
        plt.ylabel('y')
        d = alpha*cur_f
        de = abs(cur_f - prev_f)
        plt.subplot(3, 1, 1)
        img.image(get_fame(plant_x, plant_y, e=cur_f, px=k, py=cur_f, n=i, k=k, b=b, title="Оптимизация k", xlabel='K'))
        if de < e:
            break
        k += d
        prev_f = cur_f
        cur_f = opt_func(x, y, k, b)
        
        sleep(0.25)
    
    prev_f = opt_func(x, y, k, b-e)
    cur_f = opt_func(x, y, k, b)
    plant_x = np.linspace(-4, 4, 50)
    plant_y = [opt_func(x, y, k, b) for b in plant_x]
    e = 0.05
    alpha = 0.005
    for i in range(500):
        plt.figure(figsize=(8, 10))
        plt.subplot(3, 1, 3)
        plt.plot(x, y, 'b.')
        plt.plot(x_, [k*p + b for p in x_])
        plt.xlabel('x')
        plt.ylabel('y')
        d = alpha*cur_f
        de = abs(cur_f - prev_f)
        plt.subplot(3, 1, 2)
        img.image(get_fame(plant_x, plant_y, e=cur_f, px=b, py=cur_f, n=i, k=k, b=b, title="Оптимизация b", xlabel='B'))
        if de < e:
            break
        b += d
        prev_f = cur_f
        cur_f = opt_func(x, y, k, b)
        
        sleep(0.25)

'''В представленом примере параметры `k` и `b` оптимизируются по очереди, что является не оптимальным подходом. На практике как правило оптимизация будет происходить одновременно по всем параметрам сразу же'''