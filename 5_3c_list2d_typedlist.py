import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass
    
    
spec = [
    # typed-list は多重にできる
    ("typedlist_2d",
     numba.types.ListType(
         numba.types.ListType(
             numba.float64))),
    ]
@jitclass(spec)
class List2dHolder():
    def __init__(self, typedlist_2d):
        self.typedlist_2d = typedlist_2d
    def modify(self):
        # 値の変更
        self.typedlist_2d[1][0] += 100.0
        # 外側に要素追加
        self.typedlist_2d.append(numba.typed.List([200.2, 300.3]))
        # 内側に要素追加
        self.typedlist_2d[1].append(400.4)
    def dump(self):
        return self.typedlist_2d


@numba.jit(nopython=True)
def use_List2dHolder(typedlist_2d):
    # jitclassに渡して格納と処理ができることを確認する
    ins = List2dHolder(typedlist_2d)
    ins.modify()
    return ins.dump()


# 入力データ
size = 3
indice = np.array([0, 1, 1 ], dtype=np.int64)
values = np.array([1.1, 2.2, 3.3 ], dtype=np.float64)

# 2重のtyped-list を生成 (型推論を補助しながら)
inner_empty = numba.typed.List([0.0])
inner_empty.clear()
typedlist_2d_on_py = numba.typed.List([inner_empty.copy() for _ in range(size)])
for i, v in zip(indice, values):
    typedlist_2d_on_py[i].append(v)

# # 注意: 誤ってtyped-listの中にPython-listを保持させるとインタプリタのメモリが不正になった
# invalid = numba.typed.List([ [1.1], [2.2, 3.3] ])


# 処理前の値を確認
print(f"before =")
print(typedlist_2d_on_py)

# データをNumba化関数を通じてjitclassへ渡して処理
res = use_List2dHolder(typedlist_2d_on_py)

# 処理後の値を確認
print(f"after: typedlist2d_on_py =")
print(typedlist_2d_on_py)
print(f"res =")
print(res)
# => コンパイルが通り、処理が値に反映された
