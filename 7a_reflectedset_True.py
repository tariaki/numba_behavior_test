import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


spec = [
    # reflected-set-(reflected=True)
    ("reflectedsetTrue", numba.types.Set(numba.int64, reflected=True)),
    ]
@jitclass(spec)
class SetHolder():
    def __init__(self, reflectedsetTrue):
        self.reflectedsetTrue = reflectedsetTrue
    def modify(self):
        self.reflectedsetTrue.add(3000)
        self.reflectedsetTrue.discard(1000)
    def dump(self):
        return self.reflectedsetTrue


@numba.jit(nopython=True)
def use_SetHolder(frompyset):

    # 各setをjitclassに渡して格納と変更ができることを確認する
    ins = SetHolder(frompyset)
    ins.modify()
    return ins.dump()


# python領域の入力データ
pyset_on_py = {1000, 2000, -3000}
print(f"""before:
    {pyset_on_py =}""")

# データをNumba化関数を通じてjitclassへ渡して処理
assert pyset_on_py  # 空のpython-setを渡すとエラーになるので確認
res = use_SetHolder(pyset_on_py)

# 処理後の値を確認
print(f"""after:
    {pyset_on_py =}
    {res =}""")
# => コンパイルが通り、処理が値に反映された
