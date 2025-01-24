import unittest
from classes.rational import Rational


class TestRational(unittest.TestCase):
    def test_initialization(self):
        # Проверка инициализации с целыми числами
        r1 = Rational(3, 4)
        self.assertEqual(r1.numerator, 3)
        self.assertEqual(r1.denominator, 4)

        # Проверка инициализации с отрицательным числителем
        r2 = Rational(-3, 4)
        self.assertEqual(r2.numerator, -3)
        self.assertEqual(r2.denominator, 4)

        # Проверка инициализации с отрицательным знаменателем
        r3 = Rational(3, -4)
        self.assertEqual(r3.numerator, -3)
        self.assertEqual(r3.denominator, 4)

        # Проверка инициализации с нулевым числителем
        r4 = Rational(0, 5)
        self.assertEqual(r4.numerator, 0)
        self.assertEqual(r4.denominator, 1)

        # Проверка инициализации с сокращением дроби
        r5 = Rational(6, 8)
        self.assertEqual(r5.numerator, 3)
        self.assertEqual(r5.denominator, 4)

        # Проверка на исключение при нулевом знаменателе
        with self.assertRaises(ValueError):
            Rational(1, 0)

    def test_addition(self):
        # Сложение двух рациональных чисел
        r1 = Rational(1, 2)
        r2 = Rational(1, 3)
        result = r1 + r2
        self.assertEqual(result.numerator, 5)
        self.assertEqual(result.denominator, 6)

        # Сложение с целым числом
        result = r1 + 2
        self.assertEqual(result.numerator, 5)
        self.assertEqual(result.denominator, 2)

    def test_subtraction(self):
        # Вычитание двух рациональных чисел
        r1 = Rational(3, 4)
        r2 = Rational(1, 4)
        result = r1 - r2
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 2)

        # Вычитание целого числа
        result = r1 - 1
        self.assertEqual(result.numerator, -1)
        self.assertEqual(result.denominator, 4)

    def test_multiplication(self):
        # Умножение двух рациональных чисел
        r1 = Rational(1, 2)
        r2 = Rational(2, 3)
        result = r1 * r2
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 3)

        # Умножение на целое число
        result = r1 * 3
        self.assertEqual(result.numerator, 3)
        self.assertEqual(result.denominator, 2)

    def test_division(self):
        # Деление двух рациональных чисел
        r1 = Rational(1, 2)
        r2 = Rational(2, 3)
        result = r1 / r2
        self.assertEqual(result.numerator, 3)
        self.assertEqual(result.denominator, 4)

        # Деление на целое число
        result = r1 / 2
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 4)

        # Деление на ноль (должно вызвать исключение)
        with self.assertRaises(ValueError):
            r1 / 0
          def test_equality(self):
        # Проверка равенства двух рациональных чисел
        r1 = Rational(1, 2)
        r2 = Rational(2, 4)
        self.assertEqual(r1, r2)

        # Проверка равенства с целым числом
        r3 = Rational(4, 2)
        self.assertEqual(r3, 2)

    def test_string_representation(self):
        # Проверка строкового представления
        r1 = Rational(3, 4)
        self.assertEqual(str(r1), "3/4")

        r2 = Rational(-3, 4)
        self.assertEqual(str(r2), "-3/4")

        r3 = Rational(6, 2)
        self.assertEqual(str(r3), "3/1")

    def test_repr_representation(self):
        # Проверка формального строкового представления
        r1 = Rational(3, 4)
        self.assertEqual(repr(r1), "Rational(3, 4)")

        r2 = Rational(-3, 4)
        self.assertEqual(repr(r2), "Rational(-3, 4)")

    def test_normalization(self):
        # Проверка нормализации дроби
        r1 = Rational(6, 8)
        self.assertEqual(r1.numerator, 3)
        self.assertEqual(r1.denominator, 4)

        r2 = Rational(-6, 8)
        self.assertEqual(r2.numerator, -3)
        self.assertEqual(r2.denominator, 4)

        r3 = Rational(6, -8)
        self.assertEqual(r3.numerator, -3)
        self.assertEqual(r3.denominator, 4)

    def test_division_by_zero(self):
        # Проверка деления на ноль
        r1 = Rational(1, 2)
        with self.assertRaises(ValueError):
            r1 / 0

        # Проверка деления на Rational(0, 1)
        r2 = Rational(0, 1)
        with self.assertRaises(ValueError):
            r1 / r2

    def test_large_numbers(self):
        # Проверка работы с большими числами
        r1 = Rational(10**18, 1)
        r2 = Rational(1, 10**18)
        result = r1 * r2
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 1)

        # Сложение больших чисел
        r3 = Rational(10**18, 1)
        r4 = Rational(10**18, 1)
        result = r3 + r4
        self.assertEqual(result.numerator, 2 * 10**18)
        self.assertEqual(result.denominator, 1)

        # Деление больших чисел
        result = r3 / r4
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 1)

    def test_small_numbers(self):
        # Проверка работы с очень маленькими числами
        r1 = Rational(1, 10**18)
        r2 = Rational(1, 10**18)
        result = r1 * r2
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 10**36)

        # Деление очень маленьких чисел
        result = r1 / r2
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 1)

    def test_normalization_with_large_numbers(self):
        # Проверка нормализации с большими числами
        r1 = Rational(2 * 10**18, 4 * 10**18)
        self.assertEqual(r1.numerator, 1)
        self.assertEqual(r1.denominator, 2)

    def test_normalization_with_small_numbers(self):
        # Проверка нормализации с очень маленькими числами
        r1 = Rational(2, 4 * 10**18)
        self.assertEqual(r1.numerator, 1)
        self.assertEqual(r1.denominator, 2 * 10**18)
