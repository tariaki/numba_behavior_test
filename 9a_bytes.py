import numba


@numba.jit(numba.int64(numba.types.Bytes(numba.uint8, 1, "C")), nopython=True)
def bytes_mod(s):
    return len(s)
bytes_mod(b"abc")

@numba.jit("int64(Bytes(uint8, 1, \"C\"))", nopython=True)
def bytes_str(s):
    return len(s)
bytes_str(b"abc")


@numba.jit(numba.int64(numba.types.ByteArray(numba.uint8, 1, "C")), nopython=True)
def bytarr_mod(s):
    return len(s)
bytarr_mod(bytearray(b"abc"))

@numba.jit("int64(ByteArray(uint8, 1, \"C\"))", nopython=True)
def bytarr_str(s):
    return len(s)
bytarr_str(bytearray(b"abc"))


@numba.jit(numba.int64(numba.types.MemoryView(numba.uint8, 1, "C", readonly=True)), nopython=True)
def memvw_mod(s):
    return len(s)
memvw_mod(memoryview(b"abc"))

@numba.jit("int64(MemoryView(uint8, 1, \"C\", readonly=True))", nopython=True)
def memvw_str(s):
    return len(s)
memvw_str(memoryview(b"abc"))

