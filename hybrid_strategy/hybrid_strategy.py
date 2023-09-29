import datetime
import sklearn
import typing as tp
import numpy as np
import pandas as pd


X_type = tp.NewType("X_type", np.ndarray)
X_row_type = tp.NewType("X_row_type", np.ndarray)
Y_type = tp.NewType("Y_type", np.array)
TS_type = tp.NewType("TS_type", pd.Series)
Model_type = tp.TypeVar("Model_type")


def read_timeseries(path_to_df: str = "train.csv") -> TS_type:
    """Функция для чтения данных и получения обучающей и тестовой выборок"""
    df = pd.read_csv(path_to_df)
    df = df[(df['store'] == 1) & (df['item'] == 1)]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    ts = df["sales"]
    train_ts = ts[:-365]
    test_ts = ts[-365:]
    return train_ts, test_ts

def extract_hybrid_strategy_features(
        timeseries: TS_type,
        model_idx: int,
        window_size: int = 7
) -> X_row_type:
    """
    Функция для получения вектора фичей согласно гибридной схеме. На вход подаётся временной ряд
    до момента T, функция выделяет из него фичи, необходимые модели под номером model_idx для
    прогноза на момент времени T

    Args:
        timeseries --- временной ряд до момента времени T (не включительно), pd.Series с датой
                       в качестве индекса
        model_idx --- индекс модели, то есть номер шага прогноза,
                      для которого нужно получить признаки, нумерация с нуля
        window_size --- количество последних значений ряда, используемых для прогноза
                        (без учёта количества прогнозов с предыдущих этапов)

    Returns:
        Одномерный вектор фичей для модели с индексом model_idx (np.array),
        чтобы сделать прогноз для момента времени T
    """
    feature_window = model_idx + window_size
    return timeseries[-feature_window:].values


def build_datasets(
        timeseries: TS_type,
        extract_features: tp.Callable[..., X_row_type],
        window_size: int,
        model_count: int
) -> tp.List[tp.Tuple[X_type, Y_type]]:
    """
    Функция для получения обучающих датасетов согласно гибридной схеме

    Args:
        timeseries --- временной ряд
        extract_features --- функция для генерации вектора фичей
        window_size --- количество последних значений ряда, используемых для прогноза
        model_count --- количество моделей, используемых для получения предскзаний

    Returns:
        Список из model_count датасетов, i-й датасет используется для обучения i-й модели
        и представляет собой пару из двумерного массива фичей и одномерного массива таргетов
    """
    datasets = []

    # YOUR CODE HERE

    for i in range(model_count):
        features = []
        labels = []
        for j in range(len(timeseries)-window_size-i):
            series = timeseries[j:window_size+j+i+1]
            X = extract_features(series[:-1], i, window_size)
            y = series[-1]
            features.append(np.array(X))
            labels.append(np.array(y))
        datasets.append((np.array(features), np.array(labels)))

    assert len(datasets) == model_count
    return datasets

def predict(
        timeseries: TS_type,
        models: tp.List[Model_type],
        extract_features: tp.Callable[..., X_row_type] = extract_hybrid_strategy_features
) -> TS_type:
    """
    Функция для получения прогноза len(models) следующих значений временного ряда

    Args:
        timeseries --- временной ряд, по которому необходимо сделать прогноз на следующие даты
        models --- список обученных моделей, i-я модель используется для получения i-го прогноза
        extract_features --- функция для генерации вектора фичей. Если вы реализуете свою функцию
                             извлечения фичей для конечной модели, передавайте этим аргументом.
                             Внутри функции predict функцию extract_features нужно вызывать только
                             с аргументами timeseries и model_idx, остальные должны быть со значениями
                             по умолчанию

    Returns:
        Прогноз len(models) следующих значений временного ряда
    """
    y_pred = []
    for i, model in enumerate(models):
        X = extract_features(timeseries, i)
        pred = model.predict([X])[0]
        y_pred.append(pred)
        new_index = timeseries.index[-1] + pd.Timedelta("1D")
        timeseries = timeseries.append(pd.Series(pred, index=[new_index]))
    return y_pred

from sklearn.linear_model import LinearRegression


def train_models(
        train_timeseries: TS_type,
        model_count: int
) -> tp.List[Model_type]:
    """
    Функция для получения обученных моделей

    Args:
        train_timeseries --- обучающий временной ряд
        model_count --- количество моделей для обучения согласно гибридной схеме.
                        Прогнозирование должно выполняться на model_count дней вперёд

    Returns:
        Список из len(datasets) обученных моделей
    """
    models = []

    datasets = build_datasets(train_timeseries, extract_hybrid_strategy_features, model_count=model_count, window_size=7)
    for model_idx in range(model_count):
        model = LinearRegression()
        X, y = datasets[model_idx]
        model.fit(X, y)
        models.append(model)

    assert len(models) == len(datasets)
    return models