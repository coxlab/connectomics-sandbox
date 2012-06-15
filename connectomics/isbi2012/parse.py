import numpy as np
from BeautifulSoup import BeautifulSoup

html = open('./isbi2012_leaderboard-2012-06-15.html').read()
soup = BeautifulSoup(html)

tdl = [e for e in soup.findAll('td')]
tdl = tdl[6:]
tdl = np.array(tdl).reshape(-1, 5)

names = []
errors = []

for name, number, rand, warping, pixel in tdl:
    name = name.text
    rand = float(rand.text)
    warping = float(warping.text)
    pixel = float(pixel.text)
    print name, rand, warping, pixel
    names += [name]
    errors += [(rand, warping, pixel)]

errors = np.array(errors)
print errors
