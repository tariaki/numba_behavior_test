import numba


@numba.jit(nopython=True)
def type_checker_1(arg):
    return arg

# Python領域のpython-listをNumba化関数へ与える
pylist = [1.11, 2.22, 3.33]
type_checker_1(pylist)
print(f"{type_checker_1.signatures =}")
# => [(List(float64, True),)]


@numba.jit(nopython=True)
def type_checker_2(arg):
    return arg

# Numba化関数内で角括弧記法でリストを生成
@numba.jit(nopython=True)
def make_reflectedlistFalse(n):
    reflectedlistFalse = [i * 1.11 for i in range(n)]
    type_checker_2(reflectedlistFalse)
make_reflectedlistFalse(3)
print(f"{type_checker_2.signatures =}")
# => [(List(float64, False),)]


@numba.jit(nopython=True)
def type_checker_3(arg):
    return arg

# typed-list生成関数をで生成
typedlist = numba.typed.List([1.11, 2.22, 3.33])
type_checker_3(typedlist)
print(f"{type_checker_3.signatures =}")
# => [(ListType(float64),)]