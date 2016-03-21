'''
Tyler Klement: Reimplementing in Python 3.5 the code from the yhat blog:
http://blog.yhat.com/posts/logistic-regression-and-python.html

This is an implementation of logistic regression in Python using Pandas and
Statsmodels to achieve an R-style method.
'''

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

    
def cartesian(arrays, out=None):
    """
    From:
    http://stackoverflow.com/questions/1208118/
    using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:int(m),1:])
        for j in range(1, arrays[0].size):
            out[int(j*m):int((j+1)*m),1:] = out[0:int(m),1:]
    return out
    
    
def isolate_and_plot(combos, variable):
    # isolate gre and class rank
    grouped = pd.pivot_table(combos, values=['admit_pred'], index=[variable, 'position'],
                            aggfunc=np.mean)
    
    # in case you're curious as to what this looks like
    # print grouped.head()
    #                      admit_pred
    # gre        prestige            
    # 220.000000 1           0.282462
    #            2           0.169987
    #            3           0.096544
    #            4           0.079859
    # 284.444444 1           0.311718
    
    # make a plot
    colors = 'rbgyrbgy'
    for col in combos.position.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'],
                color=colors[int(col)])

    pl.xlabel(variable)
    pl.ylabel("P(admit=1)")
    pl.legend(['1', '2', '3', '4'], loc='upper left', title='Position')
    pl.title("Prob(admit=1) isolating " + variable + " and position")
    pl.show()

def main():
    dataframe = pd.read_csv("data/binary.csv")
    
    # Head from R
    print('dataframe.head() -------------------------------------------')
    print(dataframe.head())
    
    # renaming rank to position so as not to conflict with the "rank" functino from
    # pandas DataFrame
    dataframe.columns = ["admit", "gre", "gpa", "position"]
    
    print('dataframe.columns -------------------------------------------')
    print(dataframe.columns)
    
    # summary from R
    print('dataframe.describe() -------------------------------------------')
    print(dataframe.describe())
    
    # standard deviation
    print('dataframe.std() -------------------------------------------')
    print(dataframe.std())
    
    # frequency table cutting presitge and whether or not someone was admitted
    print('pd.crosstab(dataframe[\'admit\'], dataframe[\'position\'], rownames' + \
    '=[\'admit\']) -------------------------------------------')
    print(pd.crosstab(dataframe['admit'], dataframe['position'], rownames=['admit']))
    
    # plot the histogram
    dataframe.hist()
    pl.show()
    
    # dummify rank/position
    # this essentially just gives us one-hot vectors with respect to rank/position
    dummy_ranks = pd.get_dummies(dataframe['position'], prefix='position')
    print('dummy_ranks.head() -------------------------------------------')
    print(dummy_ranks.head())
    
    # create a clean data frame for the regression
    # position = 1 is going to be excluded, to avoid multicollinearity which would
    # be caused by including a dummy variable for every single category
    # My thinking: someone with a rank of 1 will always be admitted, so we have no
    # reason to include that in the data
    cols_to_keep = ['admit', 'gre', 'gpa']
    data = dataframe[cols_to_keep].join(dummy_ranks.ix[:, 'position_2':])
    print('data.head() -------------------------------------------')
    print(data.head())
    
    # manually add the intercept/bias
    data['intercept'] = 1.0
    
    train_cols = data.columns[1:]
    # Index([gre, gpa, position_2, position_3, position_4], dtype=object)
    
    # LOGISTIC REGRESSION
    # sm.Logit(y_data, x_data)
    logit = sm.Logit(data['admit'], data[train_cols])
    
    # fit the model
    result = logit.fit()
    
    print('result.summary() -------------------------------------------')
    print(result.summary())
    
    # look at the confidence interval of each coeffecient
    print('result.conf_int() -------------------------------------------')
    print(result.conf_int())
    
    # odds ratios only
    print('np.exp(result.params) -------------------------------------------')
    print(np.exp(result.params))
    
    params = result.params
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    print('np.exp(conf) -------------------------------------------')
    print(np.exp(conf))
    
    # instead of generating all possible values of GRE and GPA, we're going
    # to use an evenly spaced range of 10 values from the min to the max 
    gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
    print('gres -------------------------------------------')
    print(gres)
    # array([ 220.        ,  284.44444444,  348.88888889,  413.33333333,
    #         477.77777778,  542.22222222,  606.66666667,  671.11111111,
    #         735.55555556,  800.        ])
    gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)
    print('gpas -------------------------------------------')
    print(gpas)
    # array([ 2.26      ,  2.45333333,  2.64666667,  2.84      ,  3.03333333,
    #         3.22666667,  3.42      ,  3.61333333,  3.80666667,  4.        ])
    
    
    # enumerate all possibilities
    combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))
    # recreate the dummy variables
    combos.columns = ['gre', 'gpa', 'position', 'intercept']
    dummy_ranks = pd.get_dummies(combos['position'], prefix='position')
    dummy_ranks.columns = ['position_1', 'position_2', 'position_3', 'position_4']
    
    # keep only what we need for making predictions
    cols_to_keep = ['gre', 'gpa', 'position', 'intercept']
    combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'position_2':])
    
    # make predictions on the enumerated dataset
    combos['admit_pred'] = result.predict(combos[train_cols])
    
    print('combos.head() -------------------------------------------')
    print(combos.head())
    
    isolate_and_plot(combos, 'gre')
    isolate_and_plot(combos, 'gpa')
    
if __name__ == "__main__":
    main()