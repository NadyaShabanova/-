import pandas as pd
from matplotlib import pyplot as plt
from sklearn import *
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch

class Dataset:
    # Конструктор: Инициализирует объект с путем к файлу и загружает данные в DataFrame
    def __init__(self, path: str):
        self.path = path  # путь к файлу
        self.df = pd.read_csv(self.path)  

    # Метод подготовки данных: заполняет пропущенные значения, проверяет категориальные столбцы и выполняет их кодирование
    def preparation(self, threshold_of_num_cat: int = None, strategy='Onehot'):
        self.remove_outliers()
        self.fill_missing(strategy='mean')  # Заполняет пропущенные значения средним
        categorical = self.check_categorical(threshold_of_num_cat=threshold_of_num_cat)  # Определяет категориальные столбцы
        self.eval_categorical(categorical, strategy)  # Кодирует их по выбранной стратегии

    # Метод для удаления выбросов во всех числовых столбцах
    def remove_outliers(self):
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns  # Получаем все числовые столбцы
        for column in numeric_columns:
            q = self.df[column].quantile(q=0.99)  # Находим 99-й перцентиль
            self.df = self.df[self.df[column] <= q]  # Удаляем строки с выбросами

    # Метод для заполнения пропущенных значений в числовых столбцах с выбранной стратегией (среднее, медиана, константа)
    def fill_missing(self, strategy='mean', value=None):
        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:  # Для всех числовых столбцов
            if self.df[column].isnull().sum() > 0:  # Если в столбце есть пропущенные значения
                if strategy == 'mean':  # Если стратегия - среднее
                    self.df[column].fillna(self.df[column].mean(), inplace=True)  # Заполняем пропущенные значения средним
                elif strategy == 'median':  # Если стратегия - медиана
                    self.df[column].fillna(self.df[column].median(), inplace=True)  # Заполняем пропущенные значения медианой
                elif strategy == 'constant' and value is not None:  # Если стратегия - константа
                    self.df[column].fillna(value, inplace=True)  # Заполняем пропущенные значения заданным значением

    # Метод для проверки категориальных столбцов, основываясь на уникальных значениях и типах данных
    def check_categorical(self, threshold_of_num_cat: int = None):
        threshold_of_num_cat = threshold_of_num_cat or self.df.shape[0]  # Использует заданный порог или размер DataFrame

        #функция является ли столбец категориальным
        def is_categorical(column):
            if column.dtype in ('category', 'bool'):  # тип данных 'category' или 'bool'
                return True
            elif column.dtype == 'object':  #тип данных столбца 'object' (строка)
                return len(column.astype(str).unique()) <= threshold_of_num_cat  # количество уникальных значений
            elif column.dtype == 'float64':  # Если тип данных столбца 'float64'
                return len(column.round(10).unique()) <= threshold_of_num_cat  # количество УЗ после округления
            return len(column.unique()) <= threshold_of_num_cat  # Для других типов данных: 'int64', 'float64', 'timedelta[ns]'

        categorical_names = [column for column in self.df.columns if is_categorical(column)]  # Собирает имена категориальных столбцов
        return categorical_names  

    # Метод для кодирования категориальных столбцов в соответствии с выбранной стратегией
    def eval_categorical(self, categorical_names, strategy='Onehot') -> None:
        if strategy == 'Onehot':
            encoder = preprocessing.OneHotEncoder()  # Создаем объект для OneHot кодирования
            for col in categorical_names:  # Проходим по всем категориальным столбцам
                sub = encoder.fit_transform(self.df[col].to_numpy().reshape(-1, 1))  # Применяем OneHot кодирование
                df_encoded = pd.DataFrame(sub, columns=encoder.get_feature_names_out([col]))  # Преобразуем в DataFrame
                self.df = pd.concat([self.df, df_encoded], axis=1)  # Добавляем новые столбцы в исходный DataFrame
        elif strategy == 'Label':
            encoder = preprocessing.LabelEncoder()  # Создаем объект для Label кодирования
            for col in categorical_names:  # Проходим по всем категориальным столбцам
                self.df[col + '_labeled'] = encoder.fit_transform(self.df[col].to_numpy().reshape(-1, 1))  # Применяем Label кодирование

    # Метод для отображения графиков столбцов или всех столбцов в зависимости от типа графика
    def display(self, plot_type='Hist', column=None):
        if column:  # Если задан конкретный столбец
            # Метод для отображения графика для одного столбца
            if plot_type == 'Hist':  # тип графика - гистограмма
                self.df[column].plot(kind='hist', title=f'Histogram of {column}')
            elif plot_type == 'Box':  # тип графика - boxplot
                sns.boxplot(x=self.df[column])
                plt.title(f'Boxplot of {column}')
            else:
                raise ValueError  # Если тип графика не поддерживается, выбрасываем ошибку
        else:  # Если не задан столбец
            if plot_type == 'Hist': 
                self.df.hist(figsize=(10, 8))  # Строим гистограмму для всех числовых столбцов
            elif plot_type == 'Box': 
                sns.boxplot(data=self.df.select_dtypes(include=['float64', 'int64']))  # Строим boxplot для всех числовых столбцов
                plt.title('Boxplot for all numeric columns')
            else:
                raise ValueError  # Если тип графика не поддерживается, выбрасываем ошибку
        plt.show()

    # Метод для преобразования e в тензор для выбранного фреймворка
    def transform_to_tensor(self, framework='tensorflow'):
        if framework == 'tensorflow':  
            return tf.convert_to_tensor(self.df.values)  # Преобразуем в тензор TensorFlow
        elif framework == 'pytorch': 
            return torch.tensor(self.df.values)  # Преобразуем в тензор PyTorch
        elif framework == 'numpy':  
            return self.df.values  # Преобразуем в массив NumPy

        raise ValueError  # Если выбран неподдерживаемый фреймворк, выбрасываем ошибку
# Метод для удаления выбросов в указанном столбце (удаляем значения больше 99-го перцентиля)
    def remove_outliers(self, column_name: str):
        q = self.df[column_name].quantile(q=0.99)  # Находим 99-й перцентиль
        self.df = self.df[self.df[column_name] <= q]  # Удаляем строки с выбросами
 
    # Метод для заполнения пропущенных значений в числовых столбцах с выбранной стратегией (среднее, медиана, константа)
    def fill_missing(self, strategy='mean', value=None):
        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:  # Для всех числовых столбцов
            if self.df[column].isnull().sum() > 0:  # Если в столбце есть пропущенные значения
                if strategy == 'mean':  # Если стратегия - среднее
                    self.df[column].fillna(self.df[column].mean(), inplace=True)  # Заполняем пропущенные значения средним
                elif strategy == 'median':  # Если стратегия - медиана
                    self.df[column].fillna(self.df[column].median(), inplace=True)  # Заполняем пропущенные значения медианой
                elif strategy == 'constant' and value is not None:  # Если стратегия - константа
                    self.df[column].fillna(value, inplace=True)  # Заполняем пропущенные значения заданным значением
