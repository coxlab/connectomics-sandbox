
from scipy import stats

def reprod(p_a, p_b, prob_a):
    p_k = {}

    val = ([0,1], [prob_a, 1-prob_a])
    distrib = stats.rv_discrete(name='custm', values=val)

    if p_a.keys() != p_b.keys():
        raise ValueError, p_a.keys() != p_b.keys()

    params = (p_a, p_b)
    
    for key in p_a.keys():

        # branch?
        if type(p_a[key]) == dict:
            p_k[key] = reprod(p_a[key], p_b[key], prob_a)
        # value?
        else:
            idx = distrib.rvs(size=1)
            p_k[key] = params[idx][key]
    
    return p_k
