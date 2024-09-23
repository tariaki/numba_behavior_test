import numba


@numba.jit(nopython=True)
def type_checker_1(arg):
    return arg

# Python領域でtyped-list生成関数を使用して生成
typeddict_in_py = numba.typed.Dict()
typeddict_in_py[1] = 1.11
typeddict_in_py[2] = 2.22
type_checker_1(typeddict_in_py)
print(f"{type_checker_1.signatures =}")
# => [(DictType(int64, float64),)]


@numba.jit(nopython=True)
def type_checker_2(arg):
    return arg

# Numba化関数内でtyped-list生成関数を使用して生成
@numba.jit(nopython=True)
def make_typeddict_in_numba_explicitly(n):
    typeddict_in_numba_explicitly = numba.typed.Dict()
    typeddict_in_numba_explicitly[1] = 1.11
    typeddict_in_numba_explicitly[2] = 1.11
    type_checker_2(typeddict_in_numba_explicitly)
make_typeddict_in_numba_explicitly(3)
print(f"{type_checker_2.signatures =}")
# => [(DictType(int64, float64),)]


@numba.jit(nopython=True)
def type_checker_3(arg):
    return arg

# Numba化関数内で波括弧記法で辞書を生成
@numba.jit(nopython=True)
def make_typeddict_in_numba_by_braces(n):
    typeddict_in_numba_by_braces = {i: i * 1.11 for i in range(n)}
    type_checker_3(typeddict_in_numba_by_braces)
make_typeddict_in_numba_by_braces(3)
print(f"{type_checker_3.signatures =}")
# => [(DictType(int64, float64),)]