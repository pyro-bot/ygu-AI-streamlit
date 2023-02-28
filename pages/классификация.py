from itertools import product
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

'''
# Описание задачи классификации на примере линейной регрессии и датасета цветов Ириса
'''

raw = load_iris(as_frame=True)
labels = raw.target_names
data: pd.DataFrame = raw.frame

'Дан датасет'
st.write(data)

'''**Задача**: необходимо на основе этого датасета обучить мат. модель классифицировать цветок Ириса на виды  
**Входные параметры:**
 - sepal width (cm)
 - sepal length (cm)
 - petal width (cm)
 - petal length (cm)
 
 **Возможные виды (target):**
 1. setosa (0)
 2. versicolor (1)
 3. virginica (2)  
 
 Имеет место задача мультиклассовой классификации, но так как наша мат. модель это линейная регрессия, то она в исходном виде способна выполнять только бинарную классификацию (только 2 класса).  
 Для решения этой проблемы мы будем строить не 1, а 3 модели с основной идеей: выбираем один из 3-х классов и строим модель на бинарную классификацию в ходе которой модель должна ответить `это выбранный класс или какой то другой`.
 При такой постановки задачи мы приходим к решению не одной задачи классификации, а к рещению 3-х задач бинарной классификации.  
 
 Постоим 3 классификатора:'''

X = data.drop(columns=['target'])
y = data['target']

features = ["sepal width (cm)", "sepal length (cm)", "petal width (cm)", "petal length (cm)"]
pca = PCA(n_components=2)
cnp = pca.fit_transform(data[features])

x0 = np.linspace(cnp[:, 0].min(), cnp[:, 0].max(), 5)
y0 = np.linspace(cnp[:, 1].min(), cnp[:, 1].max(), 5)
xx, yy = np.meshgrid(x0, y0)

grid = np.vstack([xx.ravel(), yy.ravel()]).T
data_pca = pca.inverse_transform(grid)


for clf in y.unique():
    lclf = labels[clf]
    with st.expander(f"Классификация: '{lclf}' против всех"):
        f'Обучим модель на данных в которых есть 2 класса: `{lclf}` и `не {lclf}`'
        model = LinearRegression()
        pca_model = LinearRegression()
        y_ = y.apply(lambda a: int(a == clf))
        mask = y_ == 1
        model.fit(X, y_)
        pca_model.fit(cnp, y_)
        pca_y_p = pca_model.predict(cnp)
        grid_y_p = pca_model.predict(grid)

        coef = model.coef_
        pca_coef = pca_model.coef_
        f'В ходе обучения мы получили ошибку = `{model.score(X, y_)}`'
        f'''Как мы помним линейная регрессия предсказывает некоторое число, а не точное значения, однако нам надо дать точный ответ, к какому из двух классов принадлежит точка.
        Для этого будем говорить, что если ответ модели больше или равен числу `0.5` то это `{lclf}` иначе `не {clf}`. Данное число мы будем называть **пороговым**  
        Однако значения `0.5` не всегда является оптимальным и имеет смысл его изменить. Однако его изменения может привести как к улучшению точности так и к ухудшению.'''
        th = st.slider(f"Пороговое значение для класса '{lclf}' = ", 0.0, 1.0, 0.5, 0.01)
        y_p = model.predict(X) >= th

        'Метрики оценки качества обучения модели'
        col1, col2, col3 = st.columns(3)
        col1.metric('Precision:', round(precision_score(y_, y_p), 3))
        col2.metric('Recall:', round(recall_score(y_, y_p), 3))
        col3.metric('F1:', round(f1_score(y_, y_p), 3))

        mask_p = y_p
        
        fig = plt.figure()
        ax = fig.gca()
        plt.plot(cnp[mask, 0], cnp[mask, 1], 'r.', )
        plt.plot(cnp[~mask, 0], cnp[~mask, 1], 'b.', )
        display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, 
                                        response=(grid_y_p >= th).astype(int).reshape(xx.shape))
        display.plot(ax=ax)
        # plt.plot(grid[:xx.shape[0], 0], pca_model.predict(grid)[:xx.shape[0]] - th)
        st.write(fig)

        fpr, tpr, th = roc_curve(y_, model.predict(X))

        xy = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'th': np.round(th, 1)})
        fig = px.line(xy, x='fpr', y='tpr', text='th')
        fig.update_traces(textposition="bottom right")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                 line = dict(shape = 'linear', color = 'rgb(255, 0, 0)', width = 2, dash = 'dot'),
                                 name=''))
        
        st.plotly_chart(fig)
        st.metric('ROC AUC', round(roc_auc_score(y_, model.predict(X)), 3))

        'В общем виде полином выглядити так:'
        st.latex((f"f(X) = %s + {round(model.intercept_, 2)}") % ' + '.join([f'{round(coef[i], 2)} * {k}' for i, k in enumerate(features)]))
