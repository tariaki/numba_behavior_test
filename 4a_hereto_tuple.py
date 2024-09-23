import numba

print(f"""{numba.typeof((1.11, 2,))
    == numba.types.Tuple((numba.float64, numba.int64,)) 
    =}""")
# => True