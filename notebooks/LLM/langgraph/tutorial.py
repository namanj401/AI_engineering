from typing import TypedDict

class Movie(TypedDict):
    name: str
    year: int

movie = Movie(name = 'name',year = 2012)

from typing import Union

def squar(x : Union[int,float])->float:
    return x * x

h= squar(1)

from typing import Optional #Can be none or a specific type
from typing import Any # Can be any

square = lambda x: x * x
nums = [1,2,3,4]
square = list(map(square,nums))
square(10)

