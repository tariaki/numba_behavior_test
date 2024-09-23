import numba


print(f"{numba.types.unicode_type == numba.types.string =}")  # True

@numba.jit(numba.int64(numba.types.unicode_type), nopython=True)
def receiver1(s):
    return len(s)
receiver1("abc")

@numba.jit(numba.int64(numba.types.string), nopython=True)
def receiver2(s):
    return len(s)
receiver2("abc")

@numba.jit("int64(unicode_type)", nopython=True)
def receiver3(s):
    return len(s)
receiver3("abc")

@numba.jit("int64(string)", nopython=True)
def receiver4(s):
    return len(s)
receiver4("abc")