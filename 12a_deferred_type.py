
# 片方向連結リストのjitclass
# from: https://github.com/numba/numba-examples/blob/master/legacy/linkedlist.py

import numpy as np
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


node_type = numba.deferred_type()  # jitclass定義に必要な型名を生成。中身は後で指定する

spec = [
    ("data", numba.int32),
    ("next", numba.optional(node_type))  # node_type で型指定
    ]
@jitclass(spec)
class LinkedNode(object):
    def __init__(self, data, next):
        self.data = data
        self.next = next
    def prepend(self, data):
        return LinkedNode(data, self)

node_type.define(LinkedNode.class_type.instance_type)  # note_type型の中身を指定

@numba.jit(nopython=True)
def make_linked_node(data):
    return LinkedNode(data, None)



@numba.jit(nopython=True)
def fill_array(arr):
    head = make_linked_node(0)
    for i in range(1, arr.size):
        head = head.prepend(i)
    c = 0
    while head is not None:
        arr[c] = head.data
        head = head.next
        c += 1

def runme(n):
    arr = np.zeros(n, dtype=np.int32)
    print("begin: n=", n)
    fill_array(arr)
    print("Result:", arr)
    # Check answer
    np.testing.assert_equal(arr, np.arange(arr.size, dtype=arr.dtype)[::-1])

def main():
    runme(10**1)
    runme(10**6)
    runme(10**7)
    # (環境によってはサイズが大きいとStackOverFlowして終了することがあるらしい？)
    # https://numba.discourse.group/t/segmentation-fault-with-jitclass-linked-list/615
    # (手元では全て正常に完走した)


main()