import math
from .rational import Rational

class Complex:
    """
    Класс Complex представляет комплексные числа.
    Все входные данные автоматически преобразуются в Rational для упрощения вычислений.
    """

    def __init__(self, real: int | Rational, imaginary: int | Rational | None = None):
        """
        Инициализирует комплексное число.

        :param real: Действительная часть комплексного числа (int или Rational).
        :param imaginary: Мнимая часть комплексного числа (int или Rational, опционально).
        :raises TypeError: Если передан неподдерживаемый тип.
        """
        # Преобразуем real в Rational
        if isinstance(real, Complex):
            self._real = real.real
            self._imaginary = real.imaginary
        else:
            self._real = self._convert_to_rational(real)
            # Преобразуем imaginary в Rational
            if imaginary is not None:
                self._imaginary = self._convert_to_rational(imaginary)
            else:
                self._imaginary = Rational(0, 1)

    @property
    def real(self) -> Rational:
        """
        Возвращает действительную часть комплексного числа.

        :return: Действительная часть (Rational).
        """
        return self._real

    @real.setter
    def real(self, value: int | Rational):
        """
        Устанавливает действительную часть комплексного числа.

        :param value: Новое значение действительной части (int или Rational).
        :raises TypeError: Если передан неподдерживаемый тип.
        """
        self._real = self._convert_to_rational(value)

    @property
    def imaginary(self) -> Rational:
        """
        Возвращает мнимую часть комплексного числа.

        :return: Мнимая часть (Rational).
        """
        return self._imaginary

    @imaginary.setter
    def imaginary(self, value: int | Rational):
        """
        Устанавливает мнимую часть комплексного числа.

        :param value: Новое значение мнимой части (int или Rational).
        :raises TypeError: Если передан неподдерживаемый тип.
        """
        self._imaginary = self._convert_to_rational(value)

    def _convert_to_rational(self, value: int | Rational) -> Rational:
        """
        Преобразует значение в Rational.

        :param value: Целое число или Rational.
        :return: Значение, преобразованное в Rational.
        :raises TypeError: Если передан неподдерживаемый тип.
        """
        if isinstance(value, int):
            return Rational(value, 1)
        elif isinstance(value, Rational):
            return value
        else:
            raise TypeError(f"Неподдерживаемый тип: {type(value)}")

    def __add__(self, other: int | Rational) -> 'Complex':
        """
        Сложение двух комплексных чисел.

        :param other: Другое комплексное число, целое число или Rational.
        :return: Новое комплексное число как результат сложения.
        :raises TypeError: Если other не поддерживается.
        """
        if isinstance(other, Complex):
            return Complex(self.real + other.real, self.imaginary + other.imaginary)
        elif isinstance(other, (int, Rational)):
            other_rational = self._convert_to_rational(other)
            return Complex(self.real + other_rational, self.imaginary)
        else:
            raise TypeError(f"Неподдерживаемый тип для сложения: {type(other)}")

    def __sub__(self, other: int | Rational) -> 'Complex':
        """
        Вычитание двух комплексных чисел.

        :param other: Другое комплексное число, целое число или Rational.
        :return: Новое комплексное число как результат вычитания.
        :raises TypeError: Если other не поддерживается.
        """
        if isinstance(other, Complex):
            return Complex(self.real - other.real, self.imaginary - other.imaginary)
        elif isinstance(other, (int, Rational)):
            other_rational = self._convert_to_rational(other)
            return Complex(self.real - other_rational, self.imaginary)
        else:
            raise TypeError(f"Неподдерживаемый тип для вычитания: {type(other)}")

    def __mul__(self, other: int | Rational) -> 'Complex':
        """
        Умножение двух комплексных чисел.

        :param other: Другое комплексное число, целое число или Rational.
        :return: Новое комплексное число как результат умножения.
        :raises TypeError: Если other не поддерживается.
        """
        if isinstance(other, Complex):
            return Complex(
                self.real * other.real - self.imaginary * other.imaginary,
                self.real * other.imaginary + self.imaginary * other.real
            )
        elif isinstance(other, (int, Rational)):
            other_rational = self._convert_to_rational(other)
            return Complex(self.real * other_rational, self.imaginary * other_rational)
        else:
            raise TypeError(f"Неподдерживаемый тип для умножения: {type(other)}")

    def __truediv__(self, other: int | Rational) -> 'Complex':
        """
        Деление двух комплексных чисел.

        :param other: Другое комплексное число, целое число или Rational.
        :return: Новое комплексное число как результат деления.
        :raises TypeError: Если other не поддерживается.
        :raises ZeroDivisionError: Если other равен нулю.
        """
        if isinstance(other, Complex):
            if other.real == Rational(0, 1) and other.imaginary == Rational(0, 1):
                raise ZeroDivisionError("Деление на ноль")
            denominator = other.real * other.real + other.imaginary * other.imaginary
            return Complex(
                (self.real * other.real + self.imaginary * other.imaginary) / denominator,
                (self.imaginary * other.real - self.real * other.imaginary) / denominator
            )
        elif isinstance(other, (int, Rational)):
            other_rational = self._convert_to_rational(other)
            if other_rational == Rational(0, 1):
                raise ZeroDivisionError("Деление на ноль")
            return Complex(self.real / other_rational, self.imaginary / other_rational)
        else:
            raise TypeError(f"Неподдерживаемый тип для деления: {type(other)}")

    def __eq__(self, other: int | Rational) -> bool:
        """
        Проверка на равенство двух комплексных чисел.

        :param other: Другое комплексное число, целое число или Rational.
        :return: True, если числа равны, иначе False.
        :raises TypeError: Если other не поддерживается.
        """
        if isinstance(other, Complex):
            return self.real == other.real and self.imaginary == other.imaginary
        elif isinstance(other, (int, Rational)):
            other_rational = self._convert_to_rational(other)
            return self.real == other_rational and self.imaginary == Rational(0, 1)
        else:
            raise TypeError(f"Неподдерживаемый тип для сравнения: {type(other)}")

    def __str__(self) -> str:
        """
        Возвращает строковое представление комплексного числа.

        :return: Строковое представление комплексного числа.
        """
        return f"{self.real} + {self.imaginary}i"

    def __repr__(self) -> str:
        """
        Возвращает формальное строковое представление комплексного числа.

        :return: Формальное строковое представление комплексного числа.
        """
        return f"Complex({self.real}, {self.imaginary})"

    def module(self) -> float:
        """
        Возвращает модуль комплексного числа.

        :return: Модуль комплексного числа.
        """
        real_value = self.real.numerator / self.real.denominator  # Преобразуем Rational в float
        imaginary_value = self.imaginary.numerator / self.imaginary.denominator  # Преобразуем Rational в float
        return math.sqrt(real_value ** 2 + imaginary_value ** 2)

    def arg(self) -> float:
        """
        Возвращает аргумент комплексного числа (в радианах).

        :return: Аргумент комплексного числа.
        """
        real_value = self.real.numerator / self.real.denominator  # Преобразуем Rational в float
        imaginary_value = self.imaginary.numerator / self.imaginary.denominator  # Преобразуем Rational в float
        return math.atan2(imaginary_value, real_value)

    def polar_form(self) -> tuple[float, float]:
        """
        Возвращает полярную форму комплексного числа (модуль и аргумент).

        :return: Кортеж (модуль, аргумент).
        """
        return self.module(), self.arg()

    def exponential_form(self) -> str:
        """
        Возвращает экспоненциальную форму комплексного числа.

        :return: Строка, представляющая экспоненциальную форму.
        """
        r, theta = self.polar_form()
        return f"{r} * exp(i * {theta})"
