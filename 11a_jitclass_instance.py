
import numba
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass


spec_item = [
    ("_name", numba.types.unicode_type),
    ("_weight", numba.float64),
    ]
@jitclass(spec_item)
class Item():
    def __init__(self, name, weight):
        self._name = name
        self._weight = weight

spec_bag = [
    ("_weight", numba.float64),
    ("_items", numba.types.List(
        Item.class_type.instance_type,  # 変数型としてItem型を指定
        reflected=False)),
    ("_total_weight", numba.float64),
    ]
@jitclass(spec_bag)
class Bag():
    def __init__(self, weight):
        self._weight = weight
        items_empty = [Item("", 0.0), ] * 0  # reflected-list + 型推論を補助
        self._items = items_empty
        self._total_weight = weight
    def isempty(self):
        return len(self._items) == 0
    def get_total_weight(self):
        return self._total_weight
    def add(self, name, weight):
        item = Item(name, weight)
        self._items.append(item)
        self._total_weight += item._weight
    def view_all(self):
        print(f"{len(self._items)} items:")
        for item in self._items:
            print(item._name, item._weight)

@numba.jit(nopython=True, cache=True)
def use_bag():
    bag = Bag(1.0)
    print("isempty?", bag.isempty())
    bag.add("pencil", 0.1)
    bag.add("notebook", 0.5)
    print("isempty? :", bag.isempty())
    bag.view_all()
    print("total_weight :", bag.get_total_weight())


use_bag()