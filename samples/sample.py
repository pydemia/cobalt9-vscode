"""A Sample Syntax Coloring.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["rangeGenerator", "Membership"]

@decorator
def rangeGenerator(number):
    n = 0
    while n < number:
        yield n
        n += 1

# Dummy Code for sample syntax coloring.
class Membership(Object):
    count = 0
    def __init__(self, name):
        self.fullname = name
        Membership.count += 1

    @staticmethod
    def introduce(self):
        print("Hello, I'm a member,", self.fullname)

    def to_pandas(self, nparray, string=None):
        # To DataFrame.
        res = pd.DataFrame(nparray)
        s = re.sub(r'\r\n|\r', r'\n', string)
        print(s)
        return res

    def introduce(self):
        print("Hello, I'm a member,", self.fullname)


bruce = Membership('Bruce Lee')    # Instance bruce
bruce.introduce()                  # Hello, I'm a member, Bruce Lee


