import pandas as pd
from matplotlib import pyplot as plt
from sklearn import *
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch


class Dataset:
    def __init__(self, path: str):
        """
        Инициализирует класс Dataset и загружает данные из CSV файла.

        :param path: str - Путь к CSV файлу.
        """
        self.df = pd.read_csv(self.path)

    def preparation(self, threshold_of_num_cat: int = None, strategy='Onehot'):
        """
        Подготовка данных: заполнение пропущенных значений и кодирование категориальных переменных.

        :param threshold_of_num_cat: int, optional - Порог для определения категориальности столбцов по количеству уникальных значений.
        :param strategy: str, optional - Стратегия кодирования категориальных данных ('Onehot' или 'Label').
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns  # Находим числовые столбцы
        for column in numeric_columns:
          self.remove_outliers(column) 
          
        self.fill_missing(strategy='mean')
        categorical = self._check_categorical(threshold_of_num_cat=threshold_of_num_cat)
        self.eval_categorical(categorical, strategy)

    def _check_categorical(self, threshold_of_num_cat: int = None):
        """
        Проверяет, какие столбцы являются категориальными, основываясь на их типах и количестве уникальных значений.

        :param threshold_of_num_cat: int, optional - Порог для количества уникальных значений, выше которого столбец не считается категориальным.
        :return: list - Список категориальных столбцов.
        """
        threshold_of_num_cat = threshold_of_num_cat or self.df.shape[0]

        def is_categorical(column):
            if column.dtype in ('category', 'bool'):
                return True
            elif column.dtype == 'object':
                return len(column.astype(str).unique()) <= threshold_of_num_cat
            elif column.dtype == 'float64':
                return len(column.round(10).unique()) <= threshold_of_num_cat
            return len(column.unique()) <= threshold_of_num_cat

        categorical_names = [column for column in self.df.columns if is_categorical(self.df[column])]
        return categorical_names

    def eval_categorical(self, categorical_names, strategy='Onehot') -> None:
        """
        Применяет выбранную стратегию кодирования к категориальным столбцам.

        :param categorical_names: list - Список категориальных столбцов.
        :param strategy: str - Стратегия кодирования ('Onehot' или 'Label').
        """
        if strategy == 'Onehot':
            encoder = preprocessing.OneHotEncoder()
            for col in categorical_names:
                sub = encoder.fit_transform(self.df[col].to_numpy().reshape(-1, 1))
                df_encoded = pd.DataFrame(sub, columns=encoder.get_feature_names_out([col]))
                self.df = pd.concat([self.df, df_encoded], axis=1)
        elif strategy == 'Label':
            encoder = preprocessing.LabelEncoder()
            for col in categorical_names:
                self.df[col + '_labeled'] = encoder.fit_transform(self.df[col].to_numpy().reshape(-1, 1))

    def display(self, plot_type='Hist', column=None):
        """
        Отображает график для одного столбца или всех числовых столбцов.

        :param plot_type: str - Тип графика ('Hist' для гистограммы, 'Box' для boxplot).
        :param column: str, optional - Столбец для отображения графика. Если не задан, отображаются графики для всех столбцов.
        """
        if column:
            self._display_column_plot(plot_type, column)
        else:
            self._display_all_columns_plot(plot_type)
        plt.show()

    def _display_column_plot(self, plot_type, column):
        """
        Отображает график для одного столбца.

        :param plot_type: str - Тип графика ('Hist' для гистограммы, 'Box' для boxplot).
        :param column: str - Столбец для отображения графика.
        """
        if plot_type == 'Hist':
            self.df[column].plot(kind='hist', title=f'Histogram of {column}')
        elif plot_type == 'Box':
            sns.boxplot(x=self.df[column])
            plt.title(f'Boxplot of {column}')
        else:
            raise ValueError

    def _display_all_columns_plot(self, plot_type):
        """
        Отображает графики для всех числовых столбцов.

        :param plot_type: str - Тип графика ('Hist' для гистограммы, 'Box' для boxplot).
        """
        if plot_type == 'Hist':
            self.df.hist(figsize=(10, 8))
        elif plot_type == 'Box':
            sns.boxplot(data=self.df.select_dtypes(include=['float64', 'int64']))
            plt.title('Boxplot for all numeric columns')
        else:
            raise ValueError

    def transform_to_tensor(self, framework='tensorflow'):
        """
        Преобразует данные в тензор для выбранного фреймворка.

        :param framework: str - Фреймворк для преобразования ('tensorflow', 'pytorch', 'numpy').
        :return: Tensor или numpy array - Данные в виде тензора.
        """
        if framework == 'tensorflow':
            return tf.convert_to_tensor(self.df.values)
        elif framework == 'pytorch':
            return torch.tensor(self.df.values)
        elif framework == 'numpy':
            return self.df.values

        raise ValueError

    def remove_outliers(self, column_name: str):
        """
        Удаляет выбросы в столбце, значения выше 95-го перцентиля.

        :param column_name: str - Имя столбца для удаления выбросов.
        """
        q = self.df[column_name].quantile(q=0.95)
        self.df = self.df[self.df[column_name] <= q]

    def fill_missing(self, strategy='mean', value=None):
        """
        Заполняет пропущенные значения в числовых столбцах.

        :param strategy: str - Стратегия заполнения ('mean', 'median', 'constant').
        :param value: float, optional - Значение для заполнения, если стратегия 'constant'.
        """
        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
            if self.df[column].isnull().sum() > 0:
                if strategy == 'mean':
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif strategy == 'constant' and value is not None:
                    self.df[column].fillna(value, inplace=True)
