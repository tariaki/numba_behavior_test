import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


spec = [
    # reflected-list-(reflected=False)
    ("reflectedlistFalse", numba.types.List(numba.float64, reflected=False)),
    ]
@jitclass(spec)
class ListHolder():
        def __init__(self, reflectedlistFalse):
            self.reflectedlistFalse = reflectedlistFalse
        def modify(self):
            self.reflectedlistFalse[1] = 100.1
            self.reflectedlistFalse.append(200.2)
        def dump(self):
            return self.reflectedlistFalse

@numba.jit(nopython=True)
def use_ListHolder(source_array1d):
    # Numba化関数内でのlist生成表記により reflected-list-(reflected=False) を生成
    reflectedlistFalse = [v for v in source_array1d]

    ins = ListHolder(reflectedlistFalse)
    ins.modify()
    return ins.dump()


# python領域の入力データ
original = [1.11, 2.22, 3.33]
as_array = np.array(original, dtype=np.float64)
print(f"""before: 
    {original =}""")

# データをNumba化関数を通じてjitclassへ渡して処理
res = use_ListHolder(as_array)

# 処理後の値を確認
print(f"""after:
    {res =}
    {type(res) =}""")

# => コンパイルが通り、処理されたリストが得られた