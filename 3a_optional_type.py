
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass

# self.value の型として int64 と None のどちらも入ることを指定
spec = [("value", numba.optional(numba.int64)), ]
@jitclass(spec)
class CheckedInt():
    def __init__(self, num):
        self.value = num
    def div(self, right):
        if self.value is None or right.value is None:
            return CheckedInt(None)
        elif right.value == 0:
            return CheckedInt(None)
        else:  # Noneでないことを確認した分岐内でint64として扱える
            return CheckedInt(self.value // right.value)

@numba.jit(nopython=True, cache=True)
def use_chekedint(given):
    x = CheckedInt(given)
    zero = CheckedInt(0)
    a = x.div(x)
    b = x.div(zero)
    c = b.div(x)
    print(x.value, zero.value, a.value, b.value, c.value)


use_chekedint(2)
# => 動く