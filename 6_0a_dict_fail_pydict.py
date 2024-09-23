import numba

@numba.jit(nopython=True)
def receiver(arg):
    return arg

pydict = {1: 1.11, 2: 2.22, 3:3.33}

try:
    res = receiver(pydict)
    print(res)
except Exception as e:
    print(type(e), e)
# => Pythonのdictは受け入れ拒否される
# <class 'numba.core.errors.TypingError'> Failed in nopython mode pipeline (step: nopython frontend)
# non-precise type pyobject
# - argument 0: Cannot determine Numba type of <class 'dict'>