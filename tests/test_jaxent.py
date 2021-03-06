from jaxent.jaxent import *
import jax.numpy as np
import numpy as onp
import pytest

#########################
## indep model tests
#########################

N_indep = 5
indep_factors = np.array([5.21594169, 5.29834249, 2.46359556, 1.35917143, 1.08797389])
indep_marg = np.array([0.0054,0.004975,0.07845,0.204375,0.252])

def test_indep_calc_marginals_ex():
    m = Indep(N_indep, factors = indep_factors)
    m.calc_p()
    model_marginals = m.calc_marginals_ex()
    assert onp.array(indep_marg)==pytest.approx(onp.array(model_marginals))

def test_indep_train_exhuastive_from_marginals(lr=1e-1):
    m = Indep(N_indep)
    m.train(indep_marg,data_kind="marginals",data_n_samp = 10000,threshold=1,lr=lr)
    assert onp.array(m.model_marginals)==pytest.approx(onp.array(indep_marg),abs=1e-2)

def test_indep_train_exhuastive_from_data():
    m = Indep(N_indep, factors = indep_factors)
    data = m.sample(10000)
    
    m2 = Indep(N_indep)
    m2.train(data,threshold=1)
    assert onp.array(m2.model_marginals)==pytest.approx(onp.array(indep_marg),abs=1e-2)

def test_indep_train_sample_from_data():
    m = Indep(N_indep, factors = indep_factors)
    data = m.sample(10000)
    
    m2 = Indep(N_indep)
    m2.train(data,threshold=1,kind='sample')
    assert onp.array(m2.model_marginals)==pytest.approx(onp.array(indep_marg),abs=1e-2)

#########################
## ising model tests
#########################

N_ising = 15
ising_marg =  np.array([5.19054281e-03,4.77974235e-03,7.86589995e-02,2.04349559e-01
        ,2.51913050e-01,2.76749966e-01,2.05422001e-01,2.36347458e-02
        ,1.42271234e-01,1.68052429e-01,3.19850778e-02,3.62099527e-03
        ,9.39401676e-03,1.30255137e-01,1.76853354e-01,1.87719383e-05
        ,4.01079618e-04,1.29574304e-03,1.93945371e-03,1.70075446e-03
        ,1.44064886e-03,1.36354176e-04,1.55041053e-03,1.06683012e-03
        ,1.18440713e-04,1.73350126e-05,3.39812570e-05,7.09920552e-04
        ,1.11522527e-03,6.98689513e-04,1.50524310e-03,2.28570327e-03
        ,1.67263533e-03,1.77055550e-03,1.63647288e-04,7.84923218e-04
        ,1.69117855e-03,5.18778286e-04,1.09377943e-05,6.42766339e-05
        ,1.80794172e-03,1.03999960e-03,2.27524568e-02,2.49705518e-02
        ,2.85152913e-02,2.56115586e-02,3.17058693e-03,1.37191096e-02
        ,1.99645369e-02,4.54777207e-03,3.05769980e-04,1.32375326e-03
        ,1.40755387e-02,1.73938332e-02,7.07760556e-02,6.68947210e-02
        ,4.91092397e-02,7.87828837e-03,3.30033937e-02,4.75491442e-02
        ,5.72989439e-03,8.55089477e-04,1.98946782e-03,3.28963624e-02
        ,4.40534887e-02,7.96478270e-02,6.91460131e-02,7.82428913e-03
        ,4.04762534e-02,6.10291950e-02,1.07264321e-02,1.28198731e-03
        ,2.98877055e-03,4.43227848e-02,4.93863463e-02,6.12596158e-02
        ,8.41764581e-03,4.25175243e-02,5.43376322e-02,1.01241386e-02
        ,1.16478614e-03,3.60196685e-03,3.99083365e-02,5.61690095e-02
        ,5.86241883e-03,4.48160154e-02,4.23007137e-02,7.86751352e-03
        ,8.85909460e-04,3.06481567e-03,3.36868904e-02,4.68774859e-02
        ,7.40005171e-03,6.13068407e-03,1.99206422e-03,8.09434810e-05
        ,3.88680702e-04,4.74176891e-03,4.99790115e-03,2.95190649e-02
        ,4.87234894e-03,6.19563707e-04,2.20253127e-03,2.15199716e-02
        ,2.86776761e-02,9.08653271e-03,5.18539482e-04,3.29627710e-03
        ,3.31846210e-02,4.06290794e-02,5.74011271e-05,3.55350132e-04
        ,5.42875824e-03,6.45285375e-03,2.26766231e-05,5.45952112e-04
        ,6.76735981e-04,2.99795454e-03,2.78657932e-03,2.68063300e-02])
ising_factors = np.array([ 5.74972501,  6.37797427,  3.05548923,  1.79285556,  1.55978429,
        1.19984083,  1.8133614 ,  4.43684202,  2.13128999,  2.1865979 ,
        3.75577753,  5.86278422,  5.48520606,  2.31020091,  1.86405072,
        0.47276141,  0.1637651 , -0.13527034, -0.50164419, -0.18588176,
       -0.2208746 ,  0.15732444, -0.89316398, -0.10758089,  0.41797001,
        0.20246437,  0.58173385,  0.0556533 , -0.1633669 , -0.27879864,
       -0.30519535, -0.6776291 , -0.17043431, -0.57481388,  0.11018722,
        0.06393559, -0.6160062 , -1.03415946,  0.57522806,  0.28528967,
       -1.1605478 , -0.0182291 , -0.37678729, -0.14460896, -0.3513953 ,
       -0.5846285 , -0.37059304, -0.09946069, -0.39062665, -0.56765219,
       -0.02180357, -0.36395234, -0.24467886, -0.1583057 , -0.49472206,
       -0.23552937, -0.12069797, -0.55793756, -0.0914707 , -0.39249547,
        0.36487994, -0.11560748,  0.22593889, -0.18448779, -0.23499352,
       -0.18284862, -0.44690791, -0.18362084, -0.06625154, -0.5199693 ,
       -0.31306729, -0.47639625, -0.0564169 , -0.37143773, -0.04081712,
       -0.04330375, -0.26161701, -0.06925505, -0.16536977, -0.12037596,
       -0.1897996 , -0.37221428, -0.08001094, -0.18813159,  0.01819428,
       -0.64426027, -0.13177226, -0.11406202, -0.15511367, -0.40325143,
       -0.22040409, -0.3460733 , -0.97470967, -0.30857973, -0.96403878,
        0.12678515, -0.22655032, -0.3581989 , -0.07672327, -0.20261472,
        0.04473047, -0.18742879, -0.44020146, -0.0936844 , -0.09491924,
       -0.59582097,  0.29140938, -0.76525984, -0.47852701, -0.39964533,
        0.74234358,  0.12318119, -0.15909202, -0.07533527,  0.48087171,
       -0.13397996, -0.04126408, -0.97763965, -0.50263994, -0.10838287])

def test_ising_calc_marginals_ex():
    m = Ising(N_ising, factors = ising_factors)
    m.calc_p()
    model_marginals = m.calc_marginals_ex()
    assert onp.array(ising_marg)==pytest.approx(onp.array(model_marginals))

def test_ising_train_exhuastive_from_marginals():
    m = Ising(N_ising)
    m.train(ising_marg,data_kind="marginals",data_n_samp = 10000,threshold=1)
    onp.array(m.model_marginals),onp.array(ising_marg)
    assert onp.array(m.model_marginals)==pytest.approx(onp.array(ising_marg),abs=1e-2)

def test_ising_train_exhuastive_from_data():
    m = Ising(N_ising, factors = ising_factors)
    data = m.sample(10000)
    m2 = Ising(N_ising)
    m2.train(data,threshold=1)
    assert onp.array(m2.model_marginals)==pytest.approx(onp.array(ising_marg),abs=1e-2)

def test_sample_bit_to_fix():
    m = Ising(N_ising, factors = ising_factors)
    data = m.sample(10000, bits_to_fix = [0, 1, 9], values_to_fix = [1, 0, 1])
    data_mean = data.mean(0)
    assert data_mean[0]==1 and data_mean[1]==0 and data_mean[9]==1