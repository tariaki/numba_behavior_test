import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass
    
    
spec = [
    # reflected-list-(reflected=False) は多重にできる
    ("reflectedlistFalse_2d",
     numba.types.List(
         numba.types.List(
             numba.float64,
             reflected=False),
         reflected=False)),
    ]
@jitclass(spec)
class List2dHolder():
    def __init__(self, reflectedlistFalse_2d):
        self.reflectedlistFalse_2d = reflectedlistFalse_2d
    def modify(self):
        self.reflectedlistFalse_2d[1][0] += 100.0  # 値の変更
        self.reflectedlistFalse_2d.append([200.2, 300.3])  # 外側に要素追加
        self.reflectedlistFalse_2d[1].append(400.4)  # 内側に要素追加
    def dump(self):
        return self.reflectedlistFalse_2d


@numba.jit(nopython=True)
def use_List2dHolder(source_arrays):

    # Numba化関数内でのlist生成記法により2重のreflected-list-(reflected=False) を生成
    size, indice, values = source_arrays
    reflectedlistFalse_2d = [[0.0] * 0 for _N in range(size)]  # 型推論を補助
    for i, v in zip(indice, values):
        reflectedlistFalse_2d[i].append(v)

    # jitclassに渡して格納と処理ができることを確認する
    ins = List2dHolder(reflectedlistFalse_2d)
    ins.modify()
    return ins.dump()


# 入力データ
size = 3
indice = np.array([0, 1, 1 ], dtype=np.int64)
values = np.array([1.1, 2.2, 3.3 ], dtype=np.float64)

# reflected-list-(reflected=False) はNumba化関数内で生成する必要があるため、元データをNumpy配列で渡す
source_arrays = size, indice, values

# データをNumba化関数を通じてjitclassへ渡して処理
res = use_List2dHolder(source_arrays)

# 処理後の値を確認
print(f"res =")
print(res)
# => コンパイルが通り、処理が値に反映された
