import numpy as np
import numba

print(f"{numba.typeof(3.14) == numba.float64 =}")  # True
print(f"{numba.typeof(1) == numba.int64 =}")  # True
print(f"{numba.typeof(True) == numba.boolean =}")  # True
print(f"{numba.typeof(None) == numba.void =}")  # True

print()
print(
    f"""{numba.typeof(np.ones((4,3,2,),dtype=np.float64)) 
    == numba.types.Array(numba.float64, 3, 'C')
    =}""")
# True

print()
tuple_of_4 = 1.11, 1, False, np.array([[1.11]], dtype=np.float64)
print(f"{tuple_of_4 =}")  #
print(f"""{numba.typeof(tuple_of_4)
    == numba.types.Tuple((
        numba.float64, numba.int64, numba.boolean, numba.types.Array(numba.float64, 2, 'C'),
        ))
    =}""")
# True
