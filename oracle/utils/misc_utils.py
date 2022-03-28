import argparse

# from dataclasses_serialization.json import JSONSerializer
import dataclasses
import json
import warnings
from contextlib import suppress
from copy import deepcopy
from dataclasses import MISSING, fields
from typing import Any, Dict, List, Sequence, Tuple, Union, get_args

from dacite import from_dict

_nothing = object()


def seq_of_seqs_to_tuple_of_tuples(sos: Sequence[Sequence[Any]]) -> Tuple[Tuple[Any, ...], ...]:
    "convert a list of list to a tuple of tuples"
    lot: List[Tuple[Any, ...]] = []
    for s in sos:
        lot.append(tuple(s))
    out = tuple(lot)
    return out


def only_one(iterable, default=_nothing, sentinel=_nothing):
    """
    Return the first item from iterable, if and only if iterable contains a
    single element.  Raises ValueError if iterable contains more than a
    single element.  If iterable is empty, then return default value, if
    provided.  Otherwise raises ValueError.
    """
    it = iter(iterable)

    try:
        item = next(it)
    except StopIteration:
        if default is not _nothing:
            return default
        raise ValueError("zero length sequence")

    try:
        next(it)
        if sentinel is not _nothing:
            return sentinel
        raise ValueError("there can be only one")
    except StopIteration:
        return item


class ClassPropertyDescriptor(object):
    """copied from https://stackoverflow.com/questions/5189699/how-to-make-a-class-property"""

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """
    copied from https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    similar to @property decorator but works on an uninstantiated object"""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


class DataClassMixin:
    @classproperty
    def fields(cls):
        return fields(cls)

    @classproperty
    def field_names(cls):
        return [f.name for f in cls.fields]

    @classmethod
    def get_name_to_field(cls):
        return dict(zip(cls.field_names, cls.fields))

    def __repr__(self):
        keys = ",".join(self.field_names)
        return f"{self.__class__.__name__}({keys})"

    def to_json(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def __iter__(self):
        for field_name in self.field_names:
            yield getattr(self, field_name)

    def replace(self, *args, **kwargs):
        warnings.warn("use .clone", DeprecationWarning)
        return self.clone(*args, **kwargs)

    def clone(self, *args, **kwargs):
        """works well with @dataclass(frozen=True)"""
        return dataclasses.replace(deepcopy(self), *args, **kwargs)

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace):
        """
        Instantiate this dataclass from an argparse Namespace.  See the tests for detailed example usage.
        """
        return cls.from_json(vars(args))

    @classmethod
    def add_to_argparser(cls, parser) -> None:
        """
        Adds all the fields of this dataclass to argparse.  See the tests for detailed example usage.

        parser = argparse.ArgumentParser().parse_args()
        d = DataClassMixin()
        d.add_to_argparse(parser)
        d = parser.parse_args()
        d.from_parsed_args(args)

        code adapted from
        https://github.com/mivade/argparse_dataclass/blob/master/argparse_dataclass.py
        """
        for field in cls.fields:
            name = field.name
            args = field.metadata.get("args", [f"--{name}"])
            positional = not args[0].startswith("-")
            kwargs = {"type": field.metadata.get("type", field.type), "help": field.metadata.get("help", None)}

            if field.metadata.get("args") and not positional:
                # We want to ensure that we store the argument based on the
                # name of the field and not whatever flag name was provided
                kwargs["dest"] = field.name

            if field.metadata.get("choices") is not None:
                kwargs["choices"] = field.metadata["choices"]

            if field.metadata.get("nargs") is not None:
                kwargs["nargs"] = field.metadata["nargs"]
                if field.metadata.get("type") is None:
                    # When nargs is specified, field.type should be a list,
                    # or something equivalent, like typing.List.
                    # Using it would most likely result in an error, so if the user
                    # did not specify the type of the elements within the list, we
                    # try to infer it:
                    try:
                        type_ = get_args(field.type)[0]  # get_args returns a tuple
                        kwargs["type"] = type_
                    except IndexError:
                        # get_args returned an empty tuple, type cannot be inferred
                        raise ValueError(
                            f"Cannot infer type of items in field: {name}. "
                            "Try using a parameterized type hint, or "
                            "specifying the type explicitly using "
                            "metadata['type']"
                        )

            if field.default == field.default_factory == MISSING and not positional:
                kwargs["required"] = True
            else:
                if field.default_factory != MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["default"] = field.default

            if field.type is bool:
                kwargs["action"] = "store_true"

                for key in ("type", "required"):
                    with suppress(KeyError):
                        kwargs.pop(key)

            if kwargs.get("type") == dict:
                # allows parding of string to dict, ex: --arg "{'a':1}"
                kwargs["type"] = json.loads

            if kwargs.get("default"):
                s = kwargs["help"] or ""  # help might be None
                kwargs["help"] = s + f' default: {kwargs["default"]}'

            parser.add_argument(*args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return from_dict(data_class=cls, data=data)

    # this doesn't work yet, please ignore - Erik
    # @classmethod
    # def replace_defaults(cls, **kwargs):
    #     replacement = copy(cls)
    #     fields = cls.get_name_to_field()
    #     for key, val in kwargs.items():
    #         field = fields[key]
    #         field.default = val
    #     return dataclasses.dataclass(replacement)


def merge_dicts(d1: Dict[str, Any], d2: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    """merge d1 with d2 if d2 is not None else just return d1
    >>> merge_dicts(dict(a=1), dict(a=3, b=2))
    {'a': 3, 'b': 2}
    >>> merge_dicts(dict(a=1),None)
    {'a': 1}
    """
    return {**d1, **d2} if d2 else d1


def is_unique(x: Sequence):
    s = set(x)
    return len(s) != len(x)
