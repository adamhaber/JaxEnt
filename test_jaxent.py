from jaxent import *
import pytest

def test_indep_calc_marginals_ex():
    m = Indep(15)
    factors = np.array([5.21594169, 5.29834249, 2.46359556, 1.35917143, 1.08797389,
           0.96051548, 1.35272638, 3.74455168, 1.79652173, 1.59951079,
           3.41030357, 5.71052735, 4.59259772, 1.89808825, 1.53818003])
    m.create_words()
    m.calc_p(factors)
    model_marg = m.calc_marginals_ex(m.words,m.p_model)

    res = onp.array([0.0054,0.004975,0.07845,0.204375,0.252,0.276775,0.205425,0.0231,0.142275,
     0.16805,0.031975,0.0033,0.010025,0.130325,0.1768])

    assert res==pytest.approx(onp.array(model_marg))

def test_indep_train_exhuastive_from_marginals():
    m = Indep(5)
    res = onp.array([0.0054,0.004975,0.07845,0.204375,0.252])
    m.train_exhuastive(res,data_kind="marginals",data_n_samp = 10000,threshold=0.00001)
    assert onp.array(m.model_marg)==pytest.approx(res,abs=1e-6)

def test_indep_train_exhuastive_from_data():
    m = Indep(5)
    m.factors = np.array([5.21594169, 5.29834249, 2.46359556, 1.35917143, 1.08797389])
    data = m.sample(random.PRNGKey(0),100000,m.factors)
    
    m2 = Indep(5)
    m2.train_exhuastive(data,threshold=0.0001)
    res = onp.array([0.0054,0.004975,0.07845,0.204375,0.252])
    assert onp.array(m2.model_marg)==pytest.approx(res,abs=1e-2)

def test_indep_train_sample_from_data():
    m = Indep(5)
    m.factors = np.array([5.21594169, 5.29834249, 2.46359556, 1.35917143, 1.08797389])
    data = m.sample(random.PRNGKey(0),100000,m.factors)
    
    m2 = Indep(5)
    m2.train_sample(data,threshold=10)
    res = onp.array([0.0054,0.004975,0.07845,0.204375,0.252])
    assert onp.array(m2.model_marg)==pytest.approx(res,abs=1e-2)
