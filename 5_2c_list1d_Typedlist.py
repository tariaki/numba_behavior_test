import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


spec = [
    # typed-list
    ("typedlist", numba.types.ListType(numba.float64)),
    ]
@jitclass(spec)
class ListHolder():
        def __init__(self, typedlist):
            self.typedlist = typedlist
        def modify(self):
            self.typedlist[1] = 100.1
            self.typedlist.append(200.2)
        def dump(self):
            return self.typedlist

@numba.jit(nopython=True)
def use_ListHolder(typedlist):
    ins = ListHolder(typedlist)
    ins.modify()
    return ins.dump()


# python領域の入力データ
original = [1.11, 2.22, 3.33]
# typed-listを生成
typedlist_on_py = numba.typed.List(original)
print(f"""before: 
    {typedlist_on_py =}""")

# データをNumba化関数を通じてjitclassへ渡して処理
assert typedlist_on_py._typed  # 中身の型が未決定のtyped-listを渡すとエラーになるので確認
res = use_ListHolder(typedlist_on_py)

# 処理後の値を確認
print(f"""after:
    {typedlist_on_py =}
    {res =}
    {type(res) =}""")

# => コンパイルが通り、処理が値に反映された