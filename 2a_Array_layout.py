import numpy as np
import numba

# 簡易指定はレイアウト"A"
print(f"{numba.float64[:,:,:] =}")    # Array(float64, 3, 'A', ...)
# NumPyで普通に生成したNumPy配列はレイアウト"C"
arr3d = np.ones((2,3,4) , dtype=np.float64)
print(f"{numba.typeof(arr3d) =}")    # Array(float64, 3, 'C', ...)
# "A"な型指定と"C"な型指定は同等でない
print(f"{numba.float64[:,:,:] == numba.typeof(arr3d) =}")    # False

print()
# ただし、大抵の使い方ではレイアウト"A"の型指定はレイアウト"C"のデータを受け入れる
@numba.jit(numba.float64(numba.float64[:,:,:]), nopython=True)
def func1(nda_3d):
    return nda_3d[-1, -1, -1]
print(f"{func1(arr3d) =}")
# => 動く


print()
# 例えば、NumPy配列のreshapeメソッドは型指定なしならNumba内で動作する
@numba.jit(nopython=True)
def func2nosig(nda_1d):
    nda_3d = nda_1d.reshape((2,2,2,))
    return nda_3d[-1, -1, -1]
print(f"{func2nosig(np.linspace(0.0, 1.0, 8, dtype=np.float64)) =}")  # => 動く

print()
# ところがレイアウト"A"(簡易指定)で型指定したNumPy配列をreshapeするとコンパイルエラー
try:
    @numba.jit(numba.int64(numba.float64[:]), nopython=True)
    def func2A(nda_1d):
        nda_3d = nda_1d.reshape((2,2,2,))
        return nda_3d[-1, -1, -1]
except Exception as e:
    print("func2A はコンパイルエラー: ", type(e))
    # numba 0.60.0 では TypeError
    # numba 0.57.0 では numba.core.errors.TypingError


print()
# 型指定の際にレイアウト"C"を指定できる。これは実データをtypeofしたものと等価
print(f"""{numba.float64[:, :, ::1] 
    == numba.types.Array(numba.float64, 3, "C") 
    == numba.typeof(arr3d) 
    =}""")
# =>True

# 型指定をレイアウト"C"で行うと 型指定ありでもreshapeメソッドが動作する
@numba.jit(numba.int64(numba.types.Array(numba.float64, 1, "C")), nopython=True)
def func2C(nda_1d):
    nda_3d = nda_1d.reshape((2,2,2,))
    return nda_3d[-1, -1, -1]
print(f"{func2C(np.linspace(0.0, 1.0, 8, dtype=np.float64)) =}")  # => 動く
# => 動く
