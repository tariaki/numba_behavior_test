import numba

@numba.jit(nopython=True)
def receiver(arg):
    return arg

# 2重のpython-listを用意
pylist_2d = [[1.1], [2.2, 3.3]]

try:
    # 2重のpython-listをNumba化関数に与えてみる
    res = receiver(pylist_2d)
    print(res)
except Exception as e:
    print(type(e), e)
# => 受け入れ拒否される
# <class 'TypeError'> Failed in nopython mode pipeline (step: native lowering)
# cannot reflect element of reflected container: reflected list(reflected list(float64)<iv=None>)<iv=None>
