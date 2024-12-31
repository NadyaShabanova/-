class Rational:
    '''
    This class represents rational numbers as fractions of n and m.
    '''

    def init(self, n: int | None, m: int | None):
        # Инициализация объекта
        if not isinstance(n, int):
            raise ValueError("numerator must be an integer")
        if not isinstance(m, int):
            raise ValueError("denominator must be an integer")
        self.__numerator = n
        if m == 0:
            raise ValueError("Division by zero")
        self.__denominator = m

    @property
    def numerator(self):
        # Возвращает числитель.
        return self.__numerator

    @numerator.setter
    def numerator(self, value: int | None):
        # Устанавливает числитель. Проверка, что значение — целое число.
        if not isinstance(value, int):
            raise ValueError("numerator must be an integer")
        self.__numerator = value

    @property
    def denominator(self):
        # Возвращает знаменатель.
        return self.__denominator

    @denominator.setter
    def denominator(self, value: int | None):
        # Устанавливает знаменатель. Проверка, что значение — целое число и не равно нулю.
        if not isinstance(value, int):
            raise ValueError("denominator must be an integer")
        if value != 0:
            self.__denominator = value
        else:
            raise ValueError("Division by zero")
def __add__(self, other):
        # Оператор сложения (+)
        if isinstance(other, Rational):
            return Rational(self.numerator * other.denominator + self.denominator * other.numerator,
                            self.denominator * other.denominator)
        elif isinstance(other, int):
            return Rational(self.numerator + other * self.denominator,
                            self.denominator)
        else:
            raise TypeError("other operand must be an integer or Rational")

    def __sub__(self, other):
        # Оператор вычитания (-)
        if isinstance(other, Rational):
            return Rational(self.numerator * other.denominator - self.denominator * other.numerator,
                            self.denominator * other.denominator)
        elif isinstance(other, int):
            return Rational(self.numerator - other * self.denominator,
                            self.denominator)
        else:
            raise TypeError("other operand must be an integer or Rational")

    def __mul__(self, other):
        # Оператор умножения (*)
        if isinstance(other, Rational):
            return Rational(self.numerator * other.numerator,
                            self.denominator * other.denominator)
        elif isinstance(other, int):
            return Rational(self.numerator * other, self.denominator)
        else:
            raise TypeError("other operand must be an integer or Rational")

    def __truediv__(self, other):
        # Оператор деления (/)
        if isinstance(other, Rational):
            return Rational(self.numerator * other.denominator,
                            self.denominator * other.numerator)
        elif isinstance(other, int):
            return Rational(self.numerator, self.denominator * other)
        else:
            raise TypeError("other operand must be an integer or Rational")

    def __iadd__(self, other):
        # Оператор сложения с присваиванием (+=)
        if isinstance(other, Rational):
            self.numerator = self.numerator * other.denominator + self.denominator * other.numerator
            self.denominator = self.denominator * other.denominator
        elif isinstance(other, int):
            self.numerator = self.numerator + other * self.denominator
        else:
            raise TypeError("other operand must be an integer or Rational")
        return self

    def __isub__(self, other):
        # Оператор вычитания с присваиванием (-=)
        if isinstance(other, Rational):
            self.numerator = self.numerator * other.denominator - self.denominator * other.numerator
            self.denominator = self.denominator * other.denominator
        elif isinstance(other, int):
            self.numerator = self.numerator - other * self.denominator
        else:
            raise TypeError("other operand must be an integer or Rational")
        return self
def imul(self, other):
        # Оператор умножения с присваиванием (*=)
        if isinstance(other, Rational):
            self.numerator *= other.numerator
            self.denominator *= other.denominator
        elif isinstance(other, int):
            self.numerator *= other
        else:
            raise TypeError("other operand must be an integer or Rational")
        return self

    def itruediv(self, other):
        # Оператор деления с присваиванием (/=)
        if isinstance(other, Rational):
            self.numerator *= other.denominator
            self.denominator *= other.numerator
        elif isinstance(other, int):
            self.denominator *= other
        else:
            raise TypeError("other operand must be an integer or Rational")
        return self

    def eq(self, other):
        # Оператор равенства (==)
        if isinstance(other, Rational):
            return (self.numerator * other.denominator) == (self.denominator * other.numerator)
        elif isinstance(other, int):
            return self.numerator == (other * self.denominator)
        else:
            raise TypeError("other operand must be an integer or Rational")

    def ne(self, other):
        # Оператор неравенства (!=)
        return not self == other

    def pow(self, other: int | None):
        # Оператор возведения в степень (**)
        if not isinstance(other, int):
            raise TypeError("other operand must be an integer")
        if other < 0:
            return Rational(self.denominator  (abs(other)), self.numerator  (abs(other)))
        elif other > 0:
            return Rational(self.numerator  (other), self.denominator  (other))
        else:
            return Rational(1, 1)

    def abs(self):
        # Оператор абсолютного значения (abs)
        return Rational(abs(self.numerator), abs(self.denominator))

    def neg(self):
        # Унарный минус (-)
        return Rational(-self.numerator, self.denominator)

    def str(self):
        # Строковое представление объекта
        return str(round(self.numerator / self.denominator, 10))

    def repr(self):
        # Формальное строковое представление объекта
        return f"Rational({self.numerator}, {self.denominator})"

    def lshift(self, stream):
        # Оператор вывода в поток (<<)
        stream.write(str(self))
        return stream

    def print_fraction(self):
        
        return f"Rational number: {self.numerator} / {self.denominator}"
