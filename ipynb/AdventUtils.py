from collections import Counter, defaultdict, namedtuple, deque, abc
from dataclasses import dataclass, field
from itertools import permutations, combinations, cycle, chain, islice
from itertools import count as count_from, product as cross_product
from typing import *
from statistics import mean, median
from math import ceil, floor, factorial, gcd, log, log2, log10, sqrt, inf

import matplotlib.pyplot as plt

import ast
import fractions
import functools
import heapq
import operator
import pathlib
import re
import string
import time

print("starting import")

current_year = 2021  # Subdirectory name for input files
lines = str.splitlines  # By default, split input text into lines


def paragraphs(text):
    "Split text into paragraphs"
    return text.split("\n\n")

def parse(
    day_or_text: Union[int, str],
    parser: Callable = str,
    sections: Callable = lines,
    show=8,
) -> tuple:
    """Split the input text into `sections`, and apply `parser` to each.
    The first argument is either the text itself, or the day number of a text file."""
    if isinstance(day_or_text, str) and show == 8:
        show = 0  # By default, don't show lines when parsing exampole text.
    start = time.time()
    text = get_text(day_or_text)
    show_items("Puzzle input", text.splitlines(), show)
    records = mapt(parser, sections(text.rstrip()))
    if parser != str or sections != lines:
        show_items("Parsed representation", records, show)
    return records


def get_text(day_or_text: Union[int, str]) -> str:
    """The text used as input to the puzzle: either a string or the day number,
    which denotes the file 'AOC/year/input{day}.txt'."""
    if isinstance(day_or_text, str):
        return day_or_text
    else:
        filename = f"AOC/{current_year}/input{day_or_text}.txt"
        return pathlib.Path(filename).read_text()


def show_items(source, items, show: int, hr="─" * 100):
    """Show the first few items, in a pretty format."""
    if show:
        types = Counter(map(type, items))
        counts = ", ".join(
            f'{n} {t.__name__}{"" if n == 1 else "s"}' for t, n in types.items()
        )
        print(f"{hr}\n{source} ➜ {counts}:\n{hr}")
        for line in items[:show]:
            print(truncate(line))
        if show < len(items):
            print("...")


answers = {}  # `answers` is a dict of {puzzle_number: answer}


class answer:
    """Verify that calling `code` computes the `solution` to `puzzle`.
    Record results in the dict `answers`."""

    def __init__(self, puzzle, solution, code: callable):
        self.solution, self.code = solution, code
        answers[puzzle] = self
        self.check()

    def check(self):
        """Check if the code computes the correct solution; record run time."""
        start = time.time()
        self.got = self.code()
        self.secs = time.time() - start
        self.ok = self.got == self.solution
        return self.ok

    def __repr__(self):
        """The repr of an answer shows what happened."""

        def commas(x) -> str:
            return f"{x:,d}" if is_int(x) else f"{x}"

        secs = f"{self.secs:7.4f} seconds".replace(" 0.", "  .")
        ok = "" if self.ok else f" !!!! INCORRECT !!!! Expected {commas(self.solution)}"
        return f"{secs}, answer: {commas(self.got)}{ok}"


Char = str  # Intended as the type of a one-character string
Atom = Union[str, float, int]  # The type of a string or number


def ints(text: str) -> Tuple[int]:
    """A tuple of all the integers in text, ignoring non-number characters."""
    return mapt(int, re.findall(r"-?[0-9]+", text))


def positive_ints(text: str) -> Tuple[int]:
    """A tuple of all the integers in text, ignoring non-number characters."""
    return mapt(int, re.findall(r"[0-9]+", text))


def digits(text: str) -> Tuple[int]:
    """A tuple of all the digits in text (as ints 0–9), ignoring non-digit characters."""
    return mapt(int, re.findall(r"[0-9]", text))


def words(text: str) -> Tuple[str]:
    """A tuple of all the alphabetic words in text, ignoring non-letters."""
    return tuple(re.findall(r"[a-zA-Z]+", text))


def atoms(text: str) -> Tuple[Atom]:
    """A tuple of all the atoms (numbers or identifiers) in text. Skip punctuation."""
    return mapt(atom, re.findall(r"[+-]?\d+\.?\d*|\w+", text))


def atom(text: str) -> Atom:
    """Parse text into a single float or int or str."""
    try:
        x = float(text)
        return round(x) if x.is_integer() else x
    except ValueError:
        return text.strip()


def truncate(object, width=100, ellipsis=" ...") -> str:
    """Use elipsis to truncate `str(object)` to `width` characters, if necessary."""
    string = str(object)
    return (
        string if len(string) <= width else string[: width - len(ellipsis)] + ellipsis
    )


def mapt(function: Callable, *sequences) -> tuple:
    """`map`, with the result as a tuple."""
    return tuple(map(function, *sequences))


def mapl(function: Callable, *sequences) -> list:
    """`map`, with the result as a list."""
    return list(map(function, *sequences))


def quantify(iterable, pred=bool) -> int:
    """Count the number of items in iterable for which pred is true."""
    return sum(1 for item in iterable if pred(item))


def dotproduct(vec1, vec2):
    """The dot product of two vectors."""
    return sum(map(operator.mul, vec1, vec2))


def powerset(iterable) -> Iterable[tuple]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return flatten(combinations(s, r) for r in range(len(s) + 1))


flatten = chain.from_iterable  # Yield items from each sequence in turn


def append(sequences) -> Sequence:
    "Append into a list"
    return list(flatten(sequences))


def batched(iterable, n) -> Iterable[tuple]:
    "Batch data into non-overlapping tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if batch:
            yield batch
        else:
            return


def sliding_window(sequence, n) -> Iterable[Sequence]:
    """All length-n subsequences of sequence."""
    return (sequence[i : i + n] for i in range(len(sequence) + 1 - n))


def first(iterable, default=None) -> Optional[object]:
    """The first element in an iterable, or the default if iterable is empty."""
    return next(iter(iterable), default)


def last(iterable) -> Optional[object]:
    """The last element in an iterable."""
    for item in iterable:
        pass
    return item


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def first_true(iterable, default=False):
    """Returns the first true value in the iterable.
    If no true value is found, returns `default`."""
    return next((x for x in iterable if x), default)


class multimap(defaultdict):
    """A mapping of {key: [val1, val2, ...]}."""

    def __init__(self, pairs: Iterable[tuple] = (), symmetric=False):
        """Given (key, val) pairs, return {key: [val, ...], ...}.
        If `symmetric` is True, treat (key, val) as (key, val) plus (val, key)."""
        self.default_factory = list
        for key, val in pairs:
            self[key].append(val)
            if symmetric:
                self[val].append(key)


def prod(numbers) -> float:  # Will be math.prod in Python 3.8
    """The product formed by multiplying `numbers` together."""
    result = 1
    for x in numbers:
        result *= x
    return result


def T(matrix: Sequence[Sequence]) -> List[Tuple]:
    """The transpose of a matrix: T([(1,2,3), (4,5,6)]) == [(1,4), (2,5), (3,6)]"""
    return list(zip(*matrix))


def total(counter: Counter) -> int:
    """The sum of all the counts in a Counter."""
    return sum(counter.values())


def minmax(numbers) -> Tuple[int, int]:
    """A tuple of the (minimum, maximum) of numbers."""
    numbers = list(numbers)
    return min(numbers), max(numbers)


def cover(*integers) -> range:
    """A `range` that covers all the given integers, and any in between them.
    cover(lo, hi) is an inclusive (or closed) range, equal to range(lo, hi + 1).
    The same range results from cover(hi, lo) or cover([hi, lo])."""
    if len(integers) == 1:
        integers = the(integers)
    return range(min(integers), max(integers) + 1)


def the(sequence) -> object:
    """Return the one item in a sequence. Raise error if not exactly one."""
    for i, item in enumerate(sequence, 1):
        if i > 1:
            raise ValueError(f"Expected exactly one item in the sequence.")
    return item


def split_at(sequence, i) -> Tuple[Sequence, Sequence]:
    """The sequence split into two pieces: (before position i, and i-and-after)."""
    return sequence[:i], sequence[i:]


def ignore(*args) -> None:
    "Just return None."
    return None


def is_int(x) -> bool:
    "Is x an int?"
    return isinstance(x, int)


def sign(x) -> int:
    "0, +1, or -1"
    return 0 if x == 0 else +1 if x > 0 else -1


def lcm(i, j) -> int:
    "Least common multiple"
    return i * j // gcd(i, j)


def union(sets) -> set:
    "Union of several sets"
    return set().union(*sets)


def intersection(sets):
    "Intersection of several sets; error if no sets."
    first, *rest = sets
    return set(first).intersection(*rest)


def naked_plot(points, marker="o", size=(10, 10), invert=True, square=False, **kwds):
    """Plot `points` without any axis lines or tick marks.
    Optionally specify size, whether square or not, and whether to invery y axis."""
    if size:
        plt.figure(figsize=((size, size) if is_int(size) else size))
    plt.plot(*T(points), marker, **kwds)
    if square:
        plt.axis("square")
    plt.axis("off")
    if invert:
        plt.gca().invert_yaxis()


def clock_mod(i, m) -> int:
    """i % m, but replace a result of 0 with m"""
    # This is like a clock, where 24 mod 12 is 12, not 0.
    return (i % m) or m


def invert_dict(dic) -> dict:
    """Invert a dict, e.g. {1: 'a', 2: 'b'} -> {'a': 1, 'b': 2}."""
    return {dic[x]: x for x in dic}


def walrus(name, value):
    """If you're not in 3.8, and you can't do `x := val`,
    then you can use `walrus('x', val)`."""
    globals()[name] = value
    return value


Point = Tuple[int, ...]  # Type for points
Vector = Point  # E.g., (1, 0) can be a point, or can be a direction, a Vector
Zero = (0, 0)

directions4 = East, South, West, North = ((1, 0), (0, 1), (-1, 0), (0, -1))
diagonals = SE, NE, SW, NW = ((1, 1), (1, -1), (-1, 1), (-1, -1))
directions8 = directions4 + diagonals
directions5 = directions4 + (Zero,)
directions9 = directions8 + (Zero,)
arrow_direction = {
    "^": North,
    "v": South,
    ">": East,
    "<": West,
    ".": Zero,
    "U": North,
    "D": South,
    "R": East,
    "L": West,
}


def X_(point) -> int:
    "X coordinate of a point"
    return point[0]


def Y_(point) -> int:
    "Y coordinate of a point"
    return point[1]


def Z_(point) -> int:
    "Z coordinate of a point"
    return point[2]


def Xs(points) -> Tuple[int]:
    "X coordinates of a collection of points"
    return mapt(X_, points)


def Ys(points) -> Tuple[int]:
    "Y coordinates of a collection of points"
    return mapt(Y_, points)


def Zs(points) -> Tuple[int]:
    "X coordinates of a collection of points"
    return mapt(Z_, points)


def add(p: Point, q: Point) -> Point:
    return mapt(operator.add, p, q)


def sub(p: Point, q: Point) -> Point:
    return mapt(operator.sub, p, q)


def neg(p: Point) -> Vector:
    return mapt(operator.neg, p)


def mul(p: Point, k: float) -> Vector:
    return tuple(k * c for c in p)


def distance(p: Point, q: Point) -> float:
    """Euclidean (L2) distance between two points."""
    d = sum((pi - qi) ** 2 for pi, qi in zip(p, q)) ** 0.5
    return int(d) if d.is_integer() else d


def slide(points: Set[Point], delta: Vector) -> Set[Point]:
    """Slide all the points in the set of points by the amount delta."""
    return {add(p, delta) for p in points}


def make_turn(facing: Vector, turn: str) -> Vector:
    """Turn 90 degrees left or right. `turn` can be 'L' or 'Left' or 'R' or 'Right' or lowercase."""
    (x, y) = facing
    return (y, -x) if turn[0] in ("L", "l") else (-y, x)


# Profiling found that `add` and `taxi_distance` were speed bottlenecks;
# I define below versions that are specialized for 2D points only.


def add2(p: Point, q: Point) -> Point:
    """Specialized version of point addition for 2D Points only. Faster."""
    return (p[0] + q[0], p[1] + q[1])


def taxi_distance(p: Point, q: Point) -> int:
    """Manhattan (L1) distance between two 2D Points."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


class Grid(dict):
    """A 2D grid, implemented as a mapping of {(x, y): cell_contents}."""

    def __init__(self, grid=(), directions=directions4, skip=(), default=KeyError):
        """Initialize with either (e.g.) `Grid({(0, 0): '#', (1, 0): '.', ...})`, or
        `Grid(["#..", "..#"]) or `Grid("#..\n..#")`."""
        self.directions = directions
        self.default = default
        if isinstance(grid, abc.Mapping):
            self.update(grid)
        else:
            if isinstance(grid, str):
                grid = grid.splitlines()
            self.update(
                {
                    (x, y): val
                    for y, row in enumerate(grid)
                    for x, val in enumerate(row)
                    if val not in skip
                }
            )

    def __missing__(self, point):
        """If asked for a point off the grid, either return default or raise error."""
        if self.default == KeyError:
            raise KeyError(point)
        else:
            return self.default

    def copy(self):
        return Grid(self, directions=self.directions, default=self.default)

    def neighbors(self, point) -> List[Point]:
        """Points on the grid that neighbor `point`."""
        return [
            add2(point, Δ)
            for Δ in self.directions
            if add2(point, Δ) in self or self.default != KeyError
        ]

    def neighbor_contents(self, point) -> Iterable:
        """The contents of the neighboring points."""
        return (self[p] for p in self.neighbors(point))

    def to_rows(self, xrange=None, yrange=None) -> List[List[object]]:
        """The contents of the grid, as a rectangular list of lists.
        You can define a window with an xrange and yrange; or they default to the whole grid.
        """
        xrange = xrange or cover(Xs(self))
        yrange = yrange or cover(Ys(self))
        default = " " if self.default is KeyError else self.default
        return [[self.get((x, y), default) for x in xrange] for y in yrange]

    def print(self, sep="", xrange=None, yrange=None):
        """Print a representation of the grid."""
        for row in self.to_rows(xrange, yrange):
            print(*row, sep=sep)

    def plot(self, markers={"#": "s", ".": ","}, figsize=(14, 14), **kwds):
        """Plot a representation of the grid."""
        plt.figure(figsize=figsize)
        plt.gca().invert_yaxis()
        for m in markers:
            plt.plot(*T(p for p in self if self[p] == m), markers[m], **kwds)


def neighbors(point, directions=directions4) -> List[Point]:
    """Neighbors of this point, in the given directions.
    (This function can be used outside of a Grid class.)"""
    return [add(point, Δ) for Δ in direction]


cat = "".join
cache = functools.lru_cache(None)
Ø = frozenset()  # empty se
