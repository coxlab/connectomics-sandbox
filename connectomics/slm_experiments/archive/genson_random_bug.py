import genson

print genson.default_random_seed
genson.default_random_seed = 42

f = open('plos09_l3_stride_one.gson')
s = f.read()

from pprint import pprint
pprint(genson.loads(s).next())
