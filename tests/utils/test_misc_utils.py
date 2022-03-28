import argparse
from dataclasses import dataclass, field
from typing import Optional

from oracle.utils.misc_utils import DataClassMixin


def test_dataclass_mixin():
    @dataclass
    class InventoryItem(DataClassMixin):
        name: str
        unit_price: float
        quantity_on_hand: Optional[int] = None

    i = InventoryItem(name="a", unit_price=1.0, quantity_on_hand=None)
    # assert identity
    assert InventoryItem.from_json(i.to_json()) == i

    # assert we don't have to specify Optional
    assert InventoryItem.from_json({"name": "a", "unit_price": 1.0}) == InventoryItem(
        name="a", unit_price=1.0, quantity_on_hand=None
    )


def test_dataclass_mixin_argparse():
    @dataclass
    class Params(DataClassMixin):
        width: int = field(metadata=dict(help="test"))
        name: str = "joe"

    p = argparse.ArgumentParser()
    Params.add_to_argparser(p)

    args = p.parse_args("--width 1".split())
    assert Params.from_json(vars(args)) == Params(1, "joe")
