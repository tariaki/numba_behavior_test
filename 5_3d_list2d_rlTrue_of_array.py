import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass
    
    
spec = [
    # 1重のreflected-list-(reflected=True) にNumPy配列を入れて2重リストを模擬。これは動く
    ("rlTrue_of_array",
     numba.types.List(
         numba.types.Array(
             numba.float64, 1, "C"),
         reflected=True)),
    ]
@jitclass(spec)
class List2dHolder():
    def __init__(self, rlTrue_of_array):
        self.rlTrue_of_array = rlTrue_of_array
    def modify(self):
        # 値の変更
        self.rlTrue_of_array[1][0] += 100.0
        # 外側に要素追加
        self.rlTrue_of_array.append(np.array([200.2, 300.3], dtype=np.float64))
        # 内側に要素追加
        self.rlTrue_of_array[1] = np.append(self.rlTrue_of_array[1], 400.4)
            # np.appendは処理コストが重いので注意
    def dump(self):
        return self.rlTrue_of_array


@numba.jit(nopython=True)
def use_List2dHolder(frompl_of_array):
    # jitclassに渡して格納と処理ができることを確認する
    ins = List2dHolder(frompl_of_array)
    ins.modify()
    return ins.dump()


# 入力データ
size = 3
indice = np.array([0, 1, 1 ], dtype=np.int64)
values = np.array([1.1, 2.2, 3.3 ], dtype=np.float64)

# reflected-list-(reflected=False)_of_array のために pylist_of_arrayを作成
pylist_2d_temp = [[] for _i in range(size)]
for i, v in zip(indice, values):
    pylist_2d_temp[i].append(v)
pylist_of_array_on_py = [
    np.array(inner, dtype=np.float64) for inner in pylist_2d_temp
    ]

# 処理前の値を確認
print(f"before: pylist_of_array =")
print(pylist_of_array_on_py)

# データをNumba化関数を通じてjitclassへ渡して処理
res = use_List2dHolder(pylist_of_array_on_py)

# 処理後の値を確認
print(f"after: pylist_of_array =")
print(pylist_of_array_on_py)
print(f"res_rlTrue_of_array =")
print(res)
# => コンパイルが通り、処理が値に反映された
