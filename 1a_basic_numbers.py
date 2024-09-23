import numba

print(f"""{numba.typeof(3.14) 
    == numba.float64 
    == numba.f8
    == numba.types.float64
    == numba.types.f8
    =}""")
# True

print(f"""{numba.typeof(True) 
    == numba.boolean
    == numba.bool_
    == numba.b1
    == numba.types.boolean
    == numba.types.bool_
    == numba.types.b1
    =}""")
 # True