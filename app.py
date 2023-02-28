import streamlit as st
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path('.')

st.set_page_config("Стартовая страница")

'''## Введение
Данное приложение демонстрирует работу линейной регресии.  
'''
st.latex(r'f(x, k) = k_{1}x_{1} + k_{2}x_{2} +  ...+ k_{k}x_{k} + b = (\sum_{j=1}^{k} k_{j}x_{j}) + b\ \ \ \ (1)')

'''Как можно видеть линейная регрессия представляет собой уравнение в виде полинома N-ой степени  
где k - это коэфиценты уравнения и одновременно наши неизвестные, которые мы ищем, b - смещение,  а x - это наши входные данные (точки).  
Ответом на данное уравнение будет некоторое число которое мы будем интерпиритировать в зависимости от задачи сильно по разному:  
1. Для бинарной классификации - это верроятность от 0 до 1 где 0 - это принадлежность к классу "1", а 1 - это класс "2"
2. Для регрессии это численное предсказание  

Перейдем к примеру:  
Приведем уравнение `(1)` к более простому виду - уравнение прямой.  
Представим, что у нас `X` - это один параметр, тогда
'''
st.latex(r"X = \{x_1\} \\ K = \{k_1\} \\ f(X, K) = k_{1}x_{1} + b = y \ \ \ \ (2)")
'''Как можно видеть уравнение `(2)` приходмит к виду уравнения прямой'''
st.latex('y = kx + b')
'''Из этого следует, что линейная регрессия это на самом недел уравнение прямой, но для общего случая  
и на сомом деле это действительно так модель линейной регресии находит такие коэфиценты `b`, так что бы `y` стремился к 0  

## Практика'''

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
'''Слудющим шагом является построение прямой, в идеальном случае проходяшей через все точки  
Но, зная школьную программу мы помним что прямую можно провести только через 2 точки, если точек много и они не лежат строго на одной прямой, то прямую построить не получится.  
Однако выходом из данной проблемы будет построение прямой которая максимально близка ко всем точкам сразу же - это называется **аппроксимация**.  
Методом с помощью которого мы найдем такую линию называется **Метод наименьших квадратов**  
Суть метода заключается в нахождение минимума функции:'''
st.latex(r'МНК = \sum_{i=1}^{n}e^{2} = \sum_{i=1}^{n} (y_{i} - f(x_{i}))^{2} = \sum_{i=1}^{n} (y_{i} - (kx + b))^2 \rightarrow min')
'''Решить данное уравнение можно разными путями: но как правило в машинном обучение используется численная оптимизация и один из таких алгоритмов это **Метод градиентного спуска**.  
Давайте построим модель линейной регресии и найдем апроксимирующую прямую для наших данных.'''
model = LinearRegression()
model.fit(np.expand_dims(data.x, 1), data.y)
st.latex(f'y = {model.coef_[0]}x + {model.intercept_}')

'Теперь построим прямую'

x_ = np.linspace(data.x.min(), data.x.max(), 10)
y_ = model.predict(np.expand_dims(x_, 1))
fig = px.line(pd.DataFrame({'x': x_, 'y': y_}), x='x', y='y')
fig.add_trace(go.Scatter(mode='markers', x=data.x, y=data.y, name='Исходные точки'))
st.plotly_chart(fig)
'''Как можно видеть прамая проходит по центру облака точек и получается все растояние от любой точки до прямой наименьшие из возможных.  
Т.е. если бы параметры `k` и `b` были другими то среднее между всеми растояниями до прямой было бы больше чем сейчас'''
