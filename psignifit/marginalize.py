# -*- coding: utf-8 -*-
"""
get the marginal posterior distributions from your result
function [marginal,x,weight]=marginalize(result,dimension)
this function gets the marginal distribution for result
It returns 3 variables:

  marginal - the values of the marginal posterior
  x        - the gridpoints at which the posterior is sampled
  weight   - the integration weight for this point

This also works for more than one dimension in dimension to obtain
multidimensional marginals, x then becomes a cell array of the 1D ticks
of the grid.

"""
import numpy as np

def marginalize(result, dimension):
    if isinstance(dimension, (int,float)):
        if not 0 <= dimension < 5:
            raise ValueError('the dimensions you want to marginalize to should be given as a number between 0 and 4')
        d = 1
    else:
        dimension = np.asarray(dimension)
        if not np.all(0 <= dimension & dimension < 5): 
            raise ValueError('the dimensions you want to marginalize to should be given as a vector of numbers 0 to 4')
        d = len(result['X1D'])

    if d == 1 and ('marginals' in result.keys()) and len(result['marginals'])-1 >= dimension:
        marginal = result['marginals'][dimension]
        weight = result['marginalsW'][dimension]
        x = result['marginalsX'][dimension]
    elif not('Posterior' in result.keys()):
        raise ValueError('marginals cannot be computed anymore because posterior was dropped')
    elif np.shape(result['Posterior']) != np.shape(result['weight']):
        raise ValueError('dimensions mismatch in marginalization')
    else:
        if len(dimension) == 1:
            x = result['X1D'][int(dimension)][:]
        else:
            x = np.nan

        # calculate mass at each grid point
        marginal = result['weight']*result['Posterior']
        weight = result['weight']

        for i in range(0,d):
            if not(any(i == dimension)) and marginal.shape[i] > 1:
                marginal = np.sum(marginal,i, keepdims=True)
                weight = np.sum(weight,i, keepdims=True)/(np.max(result['X1D'][i])-np.min(result['X1D'][i]))
        marginal = marginal/weight

        if len(dimension) == 1:
            marginal = np.reshape(marginal, -1,1)
            weight = np.reshape(weight, -1,1)
            x = np.reshape(x,-1,1)
        else:
            marginal = np.squeeze(marginal)
            weight = np.squeeze(marginal)
            x = []
            for ix in np.sort(dimension):
                x.append(result['X1D'][ix]) 


    return (marginal, x, weight)

if __name__ == "__main__":
    import sys
    marginalize(sys.argv[1], sys.argv[2])

