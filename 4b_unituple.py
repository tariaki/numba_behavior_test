import numba

print(f"""{numba.typeof((1.11, 2.22, 3.33, 4.44, 5.55,))
    == numba.types.UniTuple(numba.float64, 5)
    =}""")
# => True