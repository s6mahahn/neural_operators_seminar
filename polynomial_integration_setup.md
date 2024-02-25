# Learning the Integration Operator for low degree polynomials
### Operator
{ f: [0,1] -> \mathbb{R} | f integrable } -> { f: [0,1] -> \mathbb{R} }

        U

{ f: [0,1] -> \mathbb{R} | f degree 4 polynomial} -> { f: [0,1] -> \mathbb{R} }

                                    f           \mapsto (t mapsto \int_0^t f(x) dx) 



### Data
**Sampling Polynomials**
- general form: f(x) = \sum_i=0^4 a_i x^i
- sample a_i ~ U([-1, 1]) (alternative: a_i ~ N(0,1))

**Antiderivative**: F(x) = \sum_i=0^4 (a_i / (i+1)) x^(i+1)

**Domain**: [0, 1]
**Grid**: evenly spaced grid in domain [0, 1] with 40 points

**Data generation steps**:
1. sample polynomial
2. calculate the integral
3. save evaluation of polynomial and the integral at all points of grid
4. save grid

**Dataset sizes**: 
*Train*: 1000
*Validation*: 200
*Test*: 200

**Note**: 
There are some hyperparameters in the described data generation you could play with! 
However, we know that this setup works, so start with the above configuration.

### Model evaluation on "In-Distribution" data
1. (if possible) compare performance of models with different hyperparameters on the test set
2. For a time series from the test set: 
    - plot polynomial and its integral 
    - evaluate your model on the (observations of the) polynomial and plot its prediction of the integral
    - does the prediction agree with the ground-truth integral?


### Model evaluation on "Out-Of-Distribution" data
1. What happens if you pass non-polynomial functions to the learned model?