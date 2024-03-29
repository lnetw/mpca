from IPython.display import HTML
import random
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go


def constant_feature_detect(data, threshold=0.99):
    """
    Нахождение константных признаков
    """
    data_copy = data.copy(deep=True)
    quasi_constant_feature = []
    for feature in data_copy.columns:
        predominant = (data_copy[feature].value_counts() / np.float(
            len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append({feature: [predominant, data_copy[feature].value_counts().index[0]]})
    # print(len(quasi_constant_feature),'константные переменные')
    return quasi_constant_feature

def hide_toggle(for_next=False):
    """
    Кнопка для скрытия частей кода
    """
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1, 2 ** 64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text
    )

    return HTML(html)

def plot_Q2(Q2, Q_005, Q_001, label):
    """
    Отрисовка Q статистики
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(x=label,
                         y=Q2,
                         showlegend=False))

    fig.add_trace(go.Scatter(x=label,
                             y=np.array([Q_005] * len(label)),
                             mode='lines',
                             line=dict(color='Black'),
                             name='95%'))

    fig.add_trace(go.Scatter(x=label,
                             y=np.array([Q_001] * len(label)),
                             mode='lines',
                             line=dict(color='Red'),
                             name='99%'))

    fig.update_layout(title_text='Q - статистика',
                      xaxis_title="Серии",
                      yaxis_title="$$E^2$$")

    return fig


def plot_T2(D, B_005, B_001, label):
    """
    Отрисовка T2 статистики
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(x=label,
                         y=np.matrix.diagonal(D),
                         showlegend=False))
    fig.add_trace(go.Scatter(x=label,
                             y=np.array([B_005] * len(label)),
                             mode='lines',
                             line=dict(color='Black'),
                             name='95%'))

    fig.add_trace(go.Scatter(x=label,
                             y=np.array([B_001] * len(label)),
                             mode='lines',
                             line=dict(color='Red'),
                             name='99%'))

    fig.update_layout(title_text='$$T^2 - статистика$$',
                      xaxis_title="Серии",
                      yaxis_title="$$Значения T^2$$")

    return fig


def plot_Tscore(t_mat, S_005, S_001, num):
    """
    Отрисока данных в новых координатах и расчет доверительных интервалов элипсоидов
    """
    t = np.linspace(0, 2 * np.pi, 100)

    fig = go.Figure()

    ##############
    for i in range(0, num):
        fig.add_trace(go.Scatter(x=t_mat[:, i],
                                 y=t_mat[:, i + 1],
                                 name='Value'))

        fig.add_trace(go.Scatter(x=np.matrix.diagonal(S_005)[i] * np.cos(t),
                                 y=np.matrix.diagonal(S_005)[i + 1] * np.sin(t),
                                 mode='lines',
                                 line=dict(color='Black'),
                                 name='95%'))

        fig.add_trace(go.Scatter(x=np.matrix.diagonal(S_001)[i] * np.cos(t),
                                 y=np.matrix.diagonal(S_001)[i + 1] * np.sin(t),
                                 mode='lines',
                                 line=dict(color='Red'),
                                 name='99%'))
    ##############
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                buttons=list([
                    dict(label="T1/T2",
                         method="update",
                         args=[{"visible": [True, True, True,
                                            False, False, False,
                                            False, False, False,
                                            False, False, False,
                                            False, False, False]},
                               {"title": "T1/T2"}]),
                    dict(label="T2/T3",
                         method="update",
                         args=[{"visible": [False, False, False,
                                            True, True, True,
                                            False, False, False,
                                            False, False, False,
                                            False, False, False]},
                               {"title": "T2/T3"}]),
                    dict(label="T3/T4",
                         method="update",
                         args=[{"visible": [False, False, False,
                                            False, False, False,
                                            True, True, True,
                                            False, False, False,
                                            False, False, False]},
                               {"title": "T3/T4"}]),
                    dict(label="T4/T5",
                         method="update",
                         args=[{"visible": [False, False, False,
                                            False, False, False,
                                            False, False, False,
                                            True, True, True,
                                            False, False, False]},
                               {"title": "T4/T5"}]),
                    dict(label="T5/T6",
                         method="update",
                         args=[{"visible": [False, False, False,
                                            False, False, False,
                                            False, False, False,
                                            False, False, False,
                                            True, True, True, ]},
                               {"title": "T5/T6"}]),
                ]),
                x=1.3,
                xanchor="center",
                y=1,
                yanchor="top"
            )
        ])
    ##########
    fig.update_layout(title_text='Доверительные интервалы по t счётам',
                      height=600, width=700,
                      xaxis_title="$$Tn$$",
                      yaxis_title="$$Tn+1$$")
    return fig


def plot_matrix(matrix, label):
    """
    Визуализация данных
    """
    fig = go.Figure()
    for i in range(matrix.shape[0]):
        fig.add_trace(go.Scatter(x=list(range(matrix.shape[1])),
                                 y=matrix,
                                 mode='lines',
                                 name=label[i]))

    fig.update_layout(title_text='Изначальные значения по процессам',
                      xaxis_title="Датчики во временном ряду одного процесса",
                      yaxis_title="Значения")
    return fig


def NIPALS(X, p_components=1):
    """
    Реализация алгоритма NIPLAS
    """
    matrix_T = np.zeros((X.shape[0], p_components))
    matrix_P = np.zeros((X.shape[1], p_components))
    X_pca = X

    for i in range(p_components):
        t_i = X_pca[:, i]
        e_t = 1
        while e_t > 10 ** -6:
            pt_i = np.divide((X_pca.T @ t_i), (t_i.T @ t_i))
            pt_i = np.divide(pt_i, np.sqrt(pt_i.T @ pt_i))
            t_i_old = t_i
            t_i = np.divide((X_pca @ pt_i), (pt_i.T @ pt_i))
            e_t = sum((t_i_old - t_i) ** 2)
        X_pca = X_pca - np.dot(t_i.reshape((X.shape[0], 1)), pt_i.reshape((1, X.shape[1])))
        matrix_T[:, i] = t_i
        matrix_P[:, i] = pt_i

    E = X - (matrix_T @ matrix_P.T)
    result = {'t': matrix_T, 'p': matrix_P, 'E': E}
    return result


def count_statistic(t, E, F_r_1_005, F_r_1_001, F_2_i_005, F_2_i_001):
    """
    Расчет всех сатитстик
    """
    # Рассчитаем Q статистику и ее доверительный интервал
    # Выпишем процессы I и ГК (главные компоненты) R
    I = t.shape[0]
    R = t.shape[1]
    # Найдем сумму квадратов ошибок
    Q = np.sum(E ** 2, axis=1)
    # Переменная V
    V = (E @ E.T) / (I - 1)
    # Найдем тетты
    tetta_1 = np.trace(V ** 1)
    tetta_2 = np.trace(V ** 2)
    tetta_3 = np.trace(V ** 3)
    # Переменная h0
    h = 1 - (2 * tetta_1 * tetta_3) / (3 * tetta_2 ** 2)
    # Выпишем кртические значения z по приложению
    Z_005 = 0.9495
    Z_001 = 0.9937
    # Найдем доверительные интервалы для Q
    Q_005 = tetta_1 * (1 - (
            (tetta_2 * h * (1 - h)) / (tetta_1 ** 2) + (Z_005 * (2 * tetta_2 * h ** 2) ** 0.5) / (tetta_1))) ** (
                    1 / h)
    Q_001 = tetta_1 * (1 - (
            (tetta_2 * h * (1 - h)) / (tetta_1 ** 2) + (Z_001 * (2 * tetta_2 * h ** 2) ** 0.5) / (tetta_1))) ** (
                    1 / h)

    # Рассчет T2 статистики
    # Найдем ковариационную матрицу и матрицу T
    ti = t
    Cov_t = np.cov(ti.T)
    # D статистика в статье в ТЗ Т2
    D = (ti @ np.linalg.inv(Cov_t) @ ti.T * I) / ((I - 1) ** 2)
    # Найдем B
    B_005 = (R / (I - R - 1) * F_r_1_005) / (1 + (R / (I - R - 1) * F_r_1_005))
    B_001 = (R / (I - R - 1) * F_r_1_001) / (1 + (R / (I - R - 1) * F_r_1_001))
    # Найдем доверительные интервалы для Т2
    T2_005 = ((Cov_t * B_005 * (I - 1) ** 2) / I) ** (1 / 2)
    T2_001 = ((Cov_t * B_001 * (I - 1) ** 2) / I) ** (1 / 2)

    ## Рассчитаем элипсоиды по t счетам
    # Найдем доверительные интервалы элипсоида
    S_005 = ((Cov_t * F_2_i_005 * 2 * (I ** 2 - 1)) / (I * (I - 2))) ** 0.5
    S_001 = ((Cov_t * F_2_i_001 * 2 * (I ** 2 - 1)) / (I * (I - 2))) ** 0.5

    return Q, Q_005, Q_001, D, T2_005, T2_001, S_005, S_001
