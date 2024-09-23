import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


spec = [
    # reflected-set-(reflected=False)
    ("reflectedsetFalse", numba.types.Set(numba.int64, reflected=False)),
    ]
@jitclass(spec)
class SetHolder():
    def __init__(self, reflectedsetFalse):
        self.reflectedsetFalse = reflectedsetFalse
    def modify(self):
        self.reflectedsetFalse.add(3000)
        self.reflectedsetFalse.discard(1000)
    def dump(self):
        return self.reflectedsetFalse


@numba.jit(nopython=True)
def use_SetHolder(source_array1d):

    # Numba化関数内でのset生成表記により reflected-set-(reflected=False) を生成
    reflectedsetFalse = set(source_array1d)

    # # Numba化関数内でのset内包表記は未対応
    # unimplemented = {v for v in source_array1d}

    # 各setをjitclassに渡して格納と変更ができることを確認する
    ins = SetHolder(reflectedsetFalse)
    ins.modify()
    return ins.dump()


# python領域の入力データ
pyset_on_py = {1000, 2000, -3000}
print(f"""before:
    {pyset_on_py =}""")
# Numba化関数へ受け渡すためにNumPy配列に変換しておく
as_array = np.array(list(pyset_on_py), dtype=np.int64)

# データをNumba化関数を通じてjitclassへ渡して処理
res = use_SetHolder(as_array)

# 処理後の値を確認
print(f"""after:
    {res =}""")
# => コンパイルが通り、処理が値に反映された
