import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass

spec = [
    ("hetero_tuple", numba.types.Tuple((
        numba.float64,
        numba.int64,
        numba.boolean))),
    ("uni_tuple", numba.types.UniTuple(numba.boolean, 5)),
    ]
@jitclass(spec)
class TupleHolder():
    def __init__(self, x):
        self.hetero_tuple = float(x), int(x), bool(x)
        self.uni_tuple = x > 1, x > 10, x > 100, x > 1000, x > 10000
    def dump(self):
        return self.hetero_tuple, self.uni_tuple

@numba.jit(nopython=True)
def use_TupleHolder(x):
    ins = TupleHolder(x)
    return ins.dump()


print(use_TupleHolder(11.1))