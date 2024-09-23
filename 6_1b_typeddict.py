import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


spec = [
    ("typeddict", numba.types.DictType(numba.int64, numba.float64)),
]
@jitclass(spec)
class DictHolder():
    def __init__(self, typeddict):
        self.typeddict = typeddict
    def modify(self):
        self.typeddict[1000] += 10.0
        self.typeddict[4000] = 4.44
        del self.typeddict[2000]
    def has(self, k):
        return k in self.typeddict
    def dump(self):
        return self.typeddict

@numba.jit(nopython=True)
def use_DictHolder(typeddict_from_py):

    # typed-dictをjitclassに渡して格納と変更ができることを確認する
    ins = DictHolder(typeddict_from_py)
    ins.modify()
    return ins.has(4000), ins.dump()


# 入力データ
pydict = {1000: 1.11, 2000: 2.22, -3000:-3.33}

# Python領域でNumbaモジュール内のtyped-dict生成関数を使用、中身の挿入、中身の挿入で型決定
typeddict_on_py = numba.typed.Dict()
for k ,v in pydict.items():
    typeddict_on_py[k] = v

# 注意: typed-dict のコンストラクタは既存辞書からの一括変換が未実装
# unimplemented: # numba.typed.Dict({1000: 1.11, 2000: 2.22, -3000:-3.33})

# Numba化関数を通じてjitclassへ渡して処理
print(f"""before:
    {typeddict_on_py =}""")
assert typeddict_on_py._typed  # 中身の型が未決定のtyped-dictを引数渡しするとエラーになるので確認
res_has, res_dump = use_DictHolder(typeddict_on_py)
print("4000 in dicts :", res_has)

# 処理後の値を確認
print(f"""after:
    {typeddict_on_py =}
    {res_dump =}""")
# => コンパイルが通り、処理が値に反映された

