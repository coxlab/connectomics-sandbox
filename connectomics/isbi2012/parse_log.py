# IPython log file

get_ipython().magic(u'run parse.py')
errors.shape
#[Out]# (57, 3)
np.corrcoef(errors[0], error[1])[0, 1]
np.corrcoef(errors[0], errors[1])[0, 1]
#[Out]# 0.89540080145699052
from scipy import stats
stats.spearmanr(errors[0], errors[1])[0]
#[Out]# 1.0
np.corrcoef(errors[0], errors[2])[0, 1]
#[Out]# 0.89538146936544216
stats.spearmanr(errors[0], errors[2])[0]
#[Out]# 1.0
stats.spearmanr(errors[:, 0], errors[:, 2])[0]
#[Out]# 0.64776042004278223
stats.spearmanr(errors[:, 0], errors[:, 1])[0]
#[Out]# 0.75938290011019649
stats.spearmanr(errors[:, 2], errors[:, 1])[0]
#[Out]# 0.83334413690283282
np.corrcoef(errors)
#[Out]# array([[ 1.        ,  0.8954008 ,  0.89538147, ...,  0.37350954,
#[Out]#          0.11053547,  0.03699459],
#[Out]#        [ 0.8954008 ,  1.        ,  1.        , ...,  0.74747662,
#[Out]#          0.54150612,  0.47808124],
#[Out]#        [ 0.89538147,  1.        ,  1.        , ...,  0.74750546,
#[Out]#          0.54154262,  0.47811937],
#[Out]#        ..., 
#[Out]#        [ 0.37350954,  0.74747662,  0.74750546, ...,  1.        ,
#[Out]#          0.96322806,  0.94080918],
#[Out]#        [ 0.11053547,  0.54150612,  0.54154262, ...,  0.96322806,
#[Out]#          1.        ,  0.99728105],
#[Out]#        [ 0.03699459,  0.47808124,  0.47811937, ...,  0.94080918,
#[Out]#          0.99728105,  1.        ]])
np.corrcoef(errors).shape
#[Out]# (57, 57)
np.corrcoef(errors.T)
#[Out]# array([[ 1.        ,  0.53859996,  0.59346489],
#[Out]#        [ 0.53859996,  1.        ,  0.74908219],
#[Out]#        [ 0.59346489,  0.74908219,  1.        ]])
