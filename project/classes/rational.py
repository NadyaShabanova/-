class Rational:
    """
    Класс Rational представляет рациональные числа в виде дробей n/m.
    Поддерживает только целые числа (int) для числителя и знаменателя.
    """

    def __init__(self, n: int, m: int):
        """
        Инициализирует объект Rational с числителем n и знаменателем m.

        :param n: Числитель рационального числа.
        :param m: Знаменатель рационального числа.
        :raises TypeError: Если n или m не являются целыми числами.
        :raises ValueError: Если m равен нулю.
        """
        if not isinstance(n, int):
            raise TypeError("Числитель должен быть целым числом (int)")
        if not isinstance(m, int):
            raise TypeError("Знаменатель должен быть целым числом (int)")
        if m == 0:
            raise ValueError("Знаменатель не может быть равен нулю")
        self.__numerator = n
        self.__denominator = m
        self.__normalize()

    def __normalize(self):
        """
        Приводит дробь к нормализованному виду (сокращает дробь и делает знаменатель положительным).
        """
        # Находим наибольший общий делитель (НОД)
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        common_divisor = gcd(abs(self.__numerator), abs(self.__denominator))
        self.__numerator //= common_divisor
        self.__denominator //= common_divisor

        # Убедимся, что знаменатель положительный
        if self.__denominator < 0:
            self.__numerator *= -1
            self.__denominator *= -1

    @property
    def numerator(self) -> int:
        """
        Возвращает числитель рационального числа.

        :return: Числитель рационального числа.
        """
        return self.__numerator

    @numerator.setter
    def numerator(self, value: int):
        """
        Устанавливает числитель рационального числа.

        :param value: Новое значение числителя (должно быть целым числом).
        :raises TypeError: Если значение не является целым числом.
        """
        if not isinstance(value, int):
            raise TypeError("Числитель должен быть целым числом (int)")
        self.__numerator = value
        self.__normalize()

    @property
    def denominator(self) -> int:
        """
        Возвращает знаменатель рационального числа.

        :return: Знаменатель рационального числа.
        """
        return self.__denominator

    @denominator.setter
    def denominator(self, value: int):
        """
        Устанавливает знаменатель рационального числа.

        :param value: Новое значение знаменателя (должно быть целым числом).
        :raises TypeError: Если значение не является целым числом.
        :raises ValueError: Если значение равно нулю.
        """
        if not isinstance(value, int):
            raise TypeError("Знаменатель должен быть целым числом (int)")
        if value == 0:
            raise ValueError("Знаменатель не может быть равен нулю")
        self.__denominator = value
        self.__normalize()

    def __add__(self, other: 'Rational | int') -> 'Rational':
        """
        Складывает текущее рациональное число с другим рациональным числом или целым числом.

        :param other: Другое рациональное число или целое число.
        :return: Новое рациональное число как результат сложения.
        :raises TypeError: Если other не является целым числом или Rational.
        """
        if isinstance(other, int):
            other = Rational(other, 1)
        elif not isinstance(other, Rational):
            raise TypeError("Операнд должен быть целым числом или Rational")
        return Rational(
            self.numerator * other.denominator + self.denominator * other.numerator,
            self.denominator * other.denominator
        )

    def __sub__(self, other: 'Rational | int') -> 'Rational':
        """
        Вычитает другое рациональное число или целое число из текущего рационального числа.

        :param other: Другое рациональное число или целое число.
        :return: Новое рациональное число как результат вычитания.
        :raises TypeError: Если other не является целым числом или Rational.
        """
        if isinstance(other, int):
            other = Rational(other, 1)
        elif not isinstance(other, Rational):
            raise TypeError("Операнд должен быть целым числом или Rational")
        return Rational(
            self.numerator * other.denominator - self.denominator * other.numerator,
            self.denominator * other.denominator
        )

    def __mul__(self, other: 'Rational | int') -> 'Rational':
        """
        Умножает текущее рациональное число на другое рациональное число или целое число.

        :param other: Другое рациональное число или целое число.
        :return: Новое рациональное число как результат умножения.
        :raises TypeError: Если other не является целым числом или Rational.
        """
        if isinstance(other, int):
            other = Rational(other, 1)
        elif not isinstance(other, Rational):
            raise TypeError("Операнд должен быть целым числом или Rational")
        return Rational(
            self.numerator * other.numerator,
            self.denominator * other.denominator
        )

    def __truediv__(self, other: 'Rational | int') -> 'Rational':
        """
        Делит текущее рациональное число на другое рациональное число или целое число.

        :param other: Другое рациональное число или целое число.
        :return: Новое рациональное число как результат деления.
        :raises TypeError: Если other не является целым числом или Rational.
        :raises ValueError: Если other равен нулю.
        """
        if isinstance(other, int):
            other = Rational(other, 1)
        elif not isinstance(other, Rational):
            raise TypeError("Операнд должен быть целым числом или Rational")
        if other.numerator == 0:
            raise ValueError("Деление на ноль")
        return Rational(
            self.numerator * other.denominator,
            self.denominator * other.numerator
        )

    def __eq__(self, other: 'Rational | int') -> bool:
        """
        Проверяет, равно ли текущее рациональное число другому рациональному числу или целому числу.

        :param other: Другое рациональное число или целое число.
        :return: True, если равны, иначе False.
        :raises TypeError: Если other не является целым числом или Rational.
        """
        if isinstance(other, int):
            other = Rational(other, 1)
        elif not isinstance(other, Rational):
            raise TypeError("Операнд должен быть целым числом или Rational")
        return (self.numerator == other.numerator) and (self.denominator == other.denominator)

    def __ne__(self, other: 'Rational | int') -> bool:
        """
        Проверяет, не равно ли текущее рациональное число другому рациональному числу или целому числу.

        :param other: Другое рациональное число или целое число.
        :return: True, если не равны, иначе False.
        :raises TypeError: Если other не является целым числом или Rational.
        """
        return not self.__eq__(other)

    def __str__(self) -> str:
        """
        Возвращает строковое представление рационального числа.

        :return: Строковое представление рационального числа.
        """
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self) -> str:
        """
        Возвращает формальное строковое представление рационального числа.

        :return: Формальное строковое представление рационального числа.
        """
        return f"Rational({self.numerator}, {self.denominator})"
