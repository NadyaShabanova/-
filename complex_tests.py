import unittest
import math
from classes.complex import Complex
from classes.rational import Rational


class TestComplex(unittest.TestCase):
    def test_initialization(self):
        # Проверка инициализации с целыми числами
        c1 = Complex(3, 4)
        self.assertEqual(c1.real, Rational(3, 1))
        self.assertEqual(c1.imaginary, Rational(4, 1))

        # Проверка инициализации с Rational
        c2 = Complex(Rational(1, 2), Rational(3, 4))
        self.assertEqual(c2.real, Rational(1, 2))
        self.assertEqual(c2.imaginary, Rational(3, 4))

        # Проверка инициализации с нулём
        c3 = Complex(0, 0)
        self.assertEqual(c3.real, Rational(0, 1))
        self.assertEqual(c3.imaginary, Rational(0, 1))

    def test_addition(self):
        # Сложение двух комплексных чисел
        c1 = Complex(1, 2)
        c2 = Complex(3, 4)
        result = c1 + c2
        self.assertEqual(result.real, Rational(4, 1))
        self.assertEqual(result.imaginary, Rational(6, 1))

        # Сложение с целым числом
        result = c1 + 5
        self.assertEqual(result.real, Rational(6, 1))
        self.assertEqual(result.imaginary, Rational(2, 1))

    def test_subtraction(self):
        # Вычитание двух комплексных чисел
        c1 = Complex(5, 6)
        c2 = Complex(1, 2)
        result = c1 - c2
        self.assertEqual(result.real, Rational(4, 1))
        self.assertEqual(result.imaginary, Rational(4, 1))

        # Вычитание целого числа
        result = c1 - 3
        self.assertEqual(result.real, Rational(2, 1))
        self.assertEqual(result.imaginary, Rational(6, 1))

    def test_multiplication(self):
        # Умножение двух комплексных чисел
        c1 = Complex(1, 2)
        c2 = Complex(3, 4)
        result = c1 * c2
        self.assertEqual(result.real, Rational(-5, 1))
        self.assertEqual(result.imaginary, Rational(10, 1))

        # Умножение на целое число
        result = c1 * 3
        self.assertEqual(result.real, Rational(3, 1))
        self.assertEqual(result.imaginary, Rational(6, 1))
def test_division(self):
        # Деление двух комплексных чисел
        c1 = Complex(1, 2)
        c2 = Complex(3, 4)
        result = c1 / c2
        self.assertEqual(result.real, Rational(11, 25))
        self.assertEqual(result.imaginary, Rational(2, 25))

        # Деление на целое число
        result = c1 / 2
        self.assertEqual(result.real, Rational(1, 2))
        self.assertEqual(result.imaginary, Rational(1, 1))

        # Деление на ноль (должно вызвать исключение)
        with self.assertRaises(ZeroDivisionError):
            c1 / 0

    def test_module(self):
        # Модуль комплексного числа
        c1 = Complex(3, 4)
        self.assertAlmostEqual(c1.module(), 5.0)

        # Модуль комплексного числа с Rational
        c2 = Complex(Rational(3, 2), Rational(4, 2))
        self.assertAlmostEqual(c2.module(), 2.5)

    def test_arg(self):
        # Аргумент комплексного числа
        c1 = Complex(1, 1)
        self.assertAlmostEqual(c1.arg(), math.pi / 4)  # 45 градусов в радианах

        # Аргумент комплексного числа с Rational
        c2 = Complex(Rational(1, 2), Rational(1, 2))
        self.assertAlmostEqual(c2.arg(), math.pi / 4)

    def test_polar_form(self):
        # Полярная форма комплексного числа
        c1 = Complex(1, 1)
        module, arg = c1.polar_form()
        self.assertAlmostEqual(module, math.sqrt(2))
        self.assertAlmostEqual(arg, math.pi / 4)

    def test_exponential_form(self):
        # Экспоненциальная форма комплексного числа
        c1 = Complex(1, 1)
        r, theta = c1.polar_form()
        expected_form = f"{r} * exp(i * {theta})"
        self.assertEqual(c1.exponential_form(), expected_form)

        # Проверка для другого комплексного числа
        c2 = Complex(0, 1)
        r, theta = c2.polar_form()
        expected_form = f"{r} * exp(i * {theta})"
        self.assertEqual(c2.exponential_form(), expected_form)

    def test_large_numbers(self):
        # Проверка на больших числах
        c1 = Complex(10**18, 10**18)
        c2 = Complex(10**18, 10**18)
        result = c1 + c2
        self.assertEqual(result.real, Rational(2 * 10**18, 1))
        self.assertEqual(result.imaginary, Rational(2 * 10**18, 1))

        result = c1 * c2
        self.assertEqual(result.real, Rational(0, 1))  # (10^18)^2 - (10^18)^2 = 0
        self.assertEqual(result.imaginary, Rational(2 * 10**36, 1))

        result = c1 / c2
        self.assertEqual(result.real, Rational(1, 1))
        self.assertEqual(result.imaginary, Rational(0, 1))

    def test_string_representation(self):
        # Проверка строкового представления
        c1 = Complex(3, 4)
        self.assertEqual(str(c1), "3/1 + 4/1i")

        c2 = Complex(Rational(1, 2), Rational(3, 4))
        self.assertEqual(str(c2), "1/2 + 3/4i")

    def test_division_by_zero(self):
        # Деление на ноль
        c1 = Complex(1, 2)
        with self.assertRaises(ZeroDivisionError):
            c1 / Complex(0, 0)

        with self.assertRaises(ZeroDivisionError):
            c1 / 0
