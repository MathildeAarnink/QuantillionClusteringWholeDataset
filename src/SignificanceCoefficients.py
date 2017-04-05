import numpy as np
from scipy.stats import chisqprob


def significanceCoefficients(X_training, logistic_function):

    predProbs = np.matrix(logistic_function.predict_proba(X_training))
    ones_column = np.ones(shape = (len(X_training),1))
    X_design = np.hstack((ones_column,X_training))
    V = np.matrix(np.zeros(shape = (X_design.shape[0], X_design.shape[0])))
    np.fill_diagonal(V,np.multiply(predProbs[:,0],predProbs[:,1]).A1)
    covlogit = np.linalg.inv(X_design.T * V * X_design)
    standard_error = np.sqrt(np.diag(covlogit))
    # print('SE', standard_error)
    logitparams = np.insert(logistic_function.coef_, 0, logistic_function.intercept_)
    # print('parameters', logitparams)
    wald_squared = (logitparams/np.sqrt(np.diag(covlogit)))**2
    #including constant (+1)
    nof_coefficients = X_training.shape[1] + 1
    degrees_of_freedom = nof_coefficients - 1
    p_values = chisqprob(wald_squared, degrees_of_freedom)
    p_values_formatted = ["%.4f" %elem for elem in p_values]
    # print('p-values', p_values_formatted)
    return logitparams, p_values


