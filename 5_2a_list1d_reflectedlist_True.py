import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


spec = [
    # reflected-list-(reflected=True)
    ("reflectedlistTrue", numba.types.List(numba.float64, reflected=True)),
    ]
@jitclass(spec)
class ListHolder():
        def __init__(self, reflectedlistTrue):
            self.reflectedlistTrue = reflectedlistTrue
        def modify(self):
            self.reflectedlistTrue[1] = 100.1
            self.reflectedlistTrue.append(200.2)
        def dump(self):
            return self.reflectedlistTrue

@numba.jit(nopython=True)
def use_ListHolder(frompylist):
    ins = ListHolder(frompylist)
    ins.modify()
    return ins.dump()


# python領域の入力データ
original = [1.11, 2.22, 3.33]
pylist_on_py = original.copy()
print(f"""before: 
    {pylist_on_py =}""")

# データをNumba化関数を通じてjitclassへ渡して処理
assert pylist_on_py  # 空のpython-listを渡すとエラーになるので確認
res = use_ListHolder(pylist_on_py)

# 処理後の値を確認
print(f"""after:
    {pylist_on_py =}
    {res =}
    {type(res) =}""")

# => コンパイルが通り、処理が値に反映された