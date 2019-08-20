
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Moving-Beyond-Linearity" data-toc-modified-id="Moving-Beyond-Linearity-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Moving Beyond Linearity</a></span><ul class="toc-item"><li><span><a href="#Polynomial-Regression" data-toc-modified-id="Polynomial-Regression-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Polynomial Regression</a></span></li><li><span><a href="#Step-Functions" data-toc-modified-id="Step-Functions-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Step Functions</a></span></li><li><span><a href="#Basis-Functions" data-toc-modified-id="Basis-Functions-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Basis Functions</a></span></li><li><span><a href="#Regression-Splines" data-toc-modified-id="Regression-Splines-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Regression Splines</a></span><ul class="toc-item"><li><span><a href="#Piecewise-Polynomials" data-toc-modified-id="Piecewise-Polynomials-7.4.1"><span class="toc-item-num">7.4.1&nbsp;&nbsp;</span>Piecewise Polynomials</a></span></li><li><span><a href="#Constraints-and-Splines" data-toc-modified-id="Constraints-and-Splines-7.4.2"><span class="toc-item-num">7.4.2&nbsp;&nbsp;</span>Constraints and Splines</a></span></li><li><span><a href="#The-Spline-Basis-Representation" data-toc-modified-id="The-Spline-Basis-Representation-7.4.3"><span class="toc-item-num">7.4.3&nbsp;&nbsp;</span>The Spline Basis Representation</a></span></li><li><span><a href="#Choosing-the-Number-and-the-Locations-of-the-Knots" data-toc-modified-id="Choosing-the-Number-and-the-Locations-of-the-Knots-7.4.4"><span class="toc-item-num">7.4.4&nbsp;&nbsp;</span>Choosing the Number and the Locations of the Knots</a></span></li><li><span><a href="#Comparison-to-Polynomial-Regression" data-toc-modified-id="Comparison-to-Polynomial-Regression-7.4.5"><span class="toc-item-num">7.4.5&nbsp;&nbsp;</span>Comparison to Polynomial Regression</a></span></li></ul></li><li><span><a href="#Smoothing-Splines" data-toc-modified-id="Smoothing-Splines-7.5"><span class="toc-item-num">7.5&nbsp;&nbsp;</span>Smoothing Splines</a></span><ul class="toc-item"><li><span><a href="#An-Overview-of-Smoothing-Splines" data-toc-modified-id="An-Overview-of-Smoothing-Splines-7.5.1"><span class="toc-item-num">7.5.1&nbsp;&nbsp;</span>An Overview of Smoothing Splines</a></span></li><li><span><a href="#Choosing-the-Smoothing-Parameter-$\lambda$" data-toc-modified-id="Choosing-the-Smoothing-Parameter-$\lambda$-7.5.2"><span class="toc-item-num">7.5.2&nbsp;&nbsp;</span>Choosing the Smoothing Parameter $\lambda$</a></span></li></ul></li><li><span><a href="#Local-Regression" data-toc-modified-id="Local-Regression-7.6"><span class="toc-item-num">7.6&nbsp;&nbsp;</span>Local Regression</a></span></li><li><span><a href="#Generalized-Additive-Models" data-toc-modified-id="Generalized-Additive-Models-7.7"><span class="toc-item-num">7.7&nbsp;&nbsp;</span>Generalized Additive Models</a></span><ul class="toc-item"><li><span><a href="#GAMs-for-Regression-Problems" data-toc-modified-id="GAMs-for-Regression-Problems-7.7.1"><span class="toc-item-num">7.7.1&nbsp;&nbsp;</span>GAMs for Regression Problems</a></span></li><li><span><a href="#GAMs-for-Classification-Problems" data-toc-modified-id="GAMs-for-Classification-Problems-7.7.2"><span class="toc-item-num">7.7.2&nbsp;&nbsp;</span>GAMs for Classification Problems</a></span></li></ul></li><li><span><a href="#Footnotes" data-toc-modified-id="Footnotes-7.8"><span class="toc-item-num">7.8&nbsp;&nbsp;</span>Footnotes</a></span></li></ul></li></ul></div>

___
# Moving Beyond Linearity
___

## Polynomial Regression

- ***Simple polynomial regression*** is a regression model which is polynomial<sup><a href='#foot54' id='ref54'>54</a></sup> in the feature variable X

$$Y = \beta_0 + \sum_{i = 1}^d \beta_iX^d$$
- The model can be fit as a simple linear regression model with predictors $X_1, \dots, X_d = X, \dots X^d$.
- It is rare to take $d \geqslant 4$ because it lead strange curves

##### Advantages

- Interpretability
- More flexibility than linear regression, can better model non-linear relationships

##### Disadvantages

- Greater flexibility can lead to overfitting (can be mitigating by keeping $d$ low)
- Imposes global structure on target function (as does linear regression)

## Step Functions

- Step functions model the target function as locally constant by converting the continuous variable $X$ into an ***ordered categorical variable***.as follows
    - Choose $K$ points $c_1, \dots, c_K \in [\min(X), \max(X)]$
    - Define $K + 1$ "dummy" variables
    \begin{align}
    C_0(X) &= I(X < c_1)\\
    C_i(X) &= I(c_i \leqslant X < c_{i+1})\qquad 1 \leqslant i \leqslant K - 1\\
    C_K(X) &= I(c_K \leqslant X)
    \end{align}
    - Fit a linear regression model to the predictors $C_1, \dots, C_K$<sup><a href='#foot55' id='ref55'>55</a>

##### Advantages

- Flexibility to model non-linear relationships
- Can model local behavior better than global models (e.g. linear and polynomial regression)

##### Disadvantages

- Locally constant assumption is strong, breakpoints in data may not be realized.

## Basis Functions

In general, we can fit a regression model

$$Y = \beta_0 + \sum_{i=1}^Kb_i(X)$$

where the $b_i(X)$ are called ***basis functions*** <sup><a href='#foot56' id='ref56'>56</a>
    

##### Advantages

Different choices of basis functions are useful for modeling different types of relationships (for example, Fourier basis functions can model periodic behavior).

##### Disadvantages

- As usual, greater flexibility can lead to overfitting
- Some choices of basis functions (i.e. basis functions which are not suited to the assumed true functional relationship) will likely have poor performance.

## Regression Splines

***Regression splines*** are a flexible (and common choice of) class of basis functions which extend both polynomial and piecewise constant basis functions.

### Piecewise Polynomials

***Piecewise polynomials*** fit separate low-degree polynomials over different regions of $X$. The points where the coefficients change are called ***knots***.

##### Advantages

- Flexibility to model non-linear relationships (as with all non-linear methods discussed in this chapter)
- Sensitivity to local behavior (less rigid than global model).

##### Disadvantages

- Overly flexible - each piece has independent degrees of freedom
- Can have unnatural breaks at knots without appropriate constraints
- Possibility of overfitting (as with all non-linear methods discussed in this chapter)

### Constraints and Splines

- To remedy overflexibility of piecewise polynomials, we can impose constraints at the knots, e.g. continuity, differentiability of various orders (smoothness).
- A ***spline*** is a piecewise degree $d$ polynomial that has continuous derivatives up to order $d-1$ at each knot (hence everywhere).

##### Advantages

- Same advantages to piecewise polynomials, while improving on the disadvantages

##### Disadvantages

- Overfitting
- Poor match to the true relationship

### The Spline Basis Representation

- Regression splines can be modeled using an appropriate basis, of which there are many choices.
- For example, we can model a $d$ degree spline with $K$ knots using ***truncated power basis*** 
    $$b_1(X), \dots, b_{K+d}(X) = x, \dots, x^d, h(X, \xi_1), \dots, h(X, \xi_K)$$
  where $\xi_i$ is the $i-th$ knot and
    $$h(X - \xi_i) =
    \begin{cases} 
        (X-\xi_i)^d & X > \xi_i\\
        0 & X \leqslant \xi_i
    \end{cases}$$
  is the ***truncated power function*** of degree $d$.

##### Advantages

Ibid.

##### Disadvantages

Beyond those mentioned above, splines can have a high variance near $\min(X), \max(X)$ (this can be overcome by using ***natural splines*** which impose boundary constraints, i.e constraints on the form of the model on $[\min(X), \xi_1]$, $[\max(X), \xi_K]$ (e.g. linearity)

### Choosing the Number and the Locations of the Knots

- In practice, we place knots in uniform fashion, e.g. by specifying the desired degrees of freedom and using software to place the knots at uniform quantiles of the data.
- The desired degrees of freedom (hence number of knots) can be obtained using cross-validation.

### Comparison to Polynomial Regression

Often gives superior results to polynomial regression -- the latter must use higher degrees (imposing global structure) while the former can increase the number of knots while leaving the degree fixed (sensitivity to local behavior) as well as varying the density of knots (i.e. placing more where the response varies rapidly, less where it is more stable)

## Smoothing Splines

### An Overview of Smoothing Splines

- A ***smoothing spline*** <sup><a href='#foot57' id='ref57'>57</a></sup> is a function

$$\hat{g}_\lambda = \underset{g}{\text{argmin}\,}\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda \int g''(t)^2\,dt$$
    where $\lambda = 0$ is a tuning parameter<sup><a href='#foot58' id='ref58'>58</a>
- $\lambda$ controls the bias-variance tradeoff. $\lambda = 0$ corresponds to the ***interpolation spline*** which fits all the data points exactly and will be thus woefull overfit. In the limit $\lambda \rightarrow \infty$, $\hat{g}_\lambda$ approaches the least squares line
- It can be show that the function $\hat{g}_\lambda$ is a piecewise cubic polynomial with knots at the unique $x_i$ and continuous first and second derivatives at the knots <sup><a href='#foot59' id='ref59'>59</a></sup>


### Choosing the Smoothing Parameter $\lambda$

- The parameter $\lambda$ controls the ***effective degrees of freedom*** $df_{\lambda}$. As $\lambda$ goes from $0 $ to $\infty$, $df_\lambda$ goes from $n$ to $2$.
- The effective degress of freedom is defined to be
$$df_\lambda = \text{trace}(S_\lambda)$$
where $S_\lambda$ is the matrix such that $\mathbf{\hat{g}}_\lambda = S_\lambda \mathbf{y}$ where $\mathbf{\hat{g}}$ is the vector of fitted values.
- $\lambda$ can be chosen by cross-validation. LOOCV is particularly efficient to compute <sup><a href='#foot60' id='ref60'>60</a></sup>

$$RSS_{cv}(\lambda) = \sum_{i=1}^n (y_i - \hat{g}_\lambda^{(-i)}(x_i))^2 = \sum_{i=1}^n\left(\frac{y_i - \hat{g}_\lambda(x_i)}{1-tr(S_{\lambda})}\right)^2 $$

##### Advantages

- Flexibility/nonlinearity
- As a shrinkage method, effective degrees of freedom are reduced, helping to balance bias-variance tradeoff and avoid overfitting.

##### Disadvantages

- As usual, flexibility can lead to overfitting

## Local Regression

- Computes the fit at a target point by regressing on nearby training observations
- Is ***memory-based***  - all the training data is necessary for computing a prediction
- In multiple linear regression, ***variable coefficient models*** fit global regression to some variables and local to others

##### Algorithm: $K$-nearest neighbors regression

Fix the parameter<sup><a href='#foot61' id='ref61'>61</a></sup> $1 \leqslant k \leqslant n$. For each $X_=x_0$:
1. Get the neighborhood $N_{i0}= \{k\ \text{closest}\ x_i\}$. 
2. Assign a weight $K_{i0} = K(x_i, x_0)$ to each point $x_i$ such that such that 
    - each point outside $x_i\notin N_{i0}$ has $K_{i0}(x_i)=0$.
    - the furthest point $x_i\in N_{i0}$ has weight zero
    - the closest point $x_i\in N_{i0}$ has the highest weight.
3. Fit a weighted least squares regression 

$$ (\hat{\beta_0}, \hat{\beta_1}) = \sum_{i=1}^nK_{i0}(y_i - \beta_0 - \beta_1 x_i)^2$$
4. Predict $\hat{f}(x_0) = \hat{\beta_0} + \hat{\beta_1}x_0$.

## Generalized Additive Models

A ***Generalized additive model*** is a model which is a sum of nonlinear functions of the individual predictors.

### GAMs for Regression Problems

- A GAM for regression  <sup><a href='#foot62' id='ref62'>62</a></sup> is a model

$$Y =\beta_0 + \sum_{j=1}^p f_j(X_j) + \epsilon$$

where the functions $f_j$ are smooth non-linear functions.

- GAMs can be used to combine methods from this chapter -- one can fit different nonlinear functions $f_j$ to the predictors $X_j$ <sup><a href='#foot63' id='ref63'>63</a></sup>
- Standard software can fit GAMs with smoothing splines via [***backfitting***](https://en.wikipedia.org/wiki/Backfitting_algorithm)

##### Advantages

- Nonlinearity hence flexibility
- Automatically introduces nonlinearity - obviates the need to experiment with different nonlinear transformations
- Interpretability/inference - the $f_j$ allow to consider the effect of each feature $X_j$ independently of the others.
- Smoothness of individual $f_j$ can be summarized via degrees of freedom.
- Represents a nice compromise betwee linear and fully non-parametric models (see [§8]()).

##### Disadvantages

- Usual disadvantages of nonlinearity
- Doesn't allow for interactions between features (this can be overcome by including nonlinear functios of the interaction terms $f(X_j,X_k)$
- The additive constraint is strong, restricts flexibility.

### GAMs for Classification Problems

GAMs can be used for classification. For example, a GAM for logistic regression is 

$$\log\left(\frac{p_k(X)}{1 - p_k(X)}\right) =\beta_0 + \sum_{j=1}^p f_j(X_j) + \epsilon$$

where $p_k(X) =\text{Pr}(Y = k\ |\ X)$.

___
## Footnotes

<p>
</p>

<div id="foot54"> 54. In statistical literature, polynomial regression is sometimes referred to as linear regression. This is because the model is linear in the population parameters $\beta_i$. 
<a href="#ref54">↩</a>
</div>

<p>
</p>

<div id="foot55"> 55. The variable $C_0(X)$ accounts for an intercept. Alternatively fit a linear model to $C_0, \dots, C_K$ with no intercept.
<a href="#ref55">↩</a>
</div>

<p>
</p>

<div id="foot56"> 56. Such a model amounts to the assumption that the target function lives in a finite-dimensional subspace of the vector space of all functions $f:X\rightarrow Y$.
<a href="#ref56">↩</a>
</div>

<p>
</p>

<div id="foot57"> 57. The function $g$ is not guaranteed to be smooth in the sense of infinitely differentiable. The penalty on the second derivative (curvature) penalizes the "roughness" or "wiggliness" of $g$, hence "smoothes out" noise in the data. Other penalties <a href="https://en.wikipedia.org/wiki/Smoothing_spline">have been used</a>
<a href="#ref57">↩</a>
</div>

<p>
</p>

<div id="foot58"> 58. A tuning parameter is also called a <a href="https://en.wikipedia.org/wiki/Hyperparameter">hyperparameter</a>
<a href="#ref58">↩</a>
</div>

<p>
</p>

<div id="foot59"> 59. Thus $\hat{g}$ is a natural cubic spline with knots at the $x_i$. However, it is not the spline one obtains in [§7.4.3](#The-Spline-Basis-Representation). It is a "shrunken" version, where $\lambda$ controls the shrinkage.
<a href="#ref59">↩</a>
</div>

<p>
</p>

<div id="foot60"> 60. Compare to a similar formula in <a href="#Leave-One-Out-Cross-Validation"> §5.1.2 </a>
<a href="#ref60">↩</a>
</div>

<p>
</p>

<div id="foot61"> 61. Our description of the algorithm deviates a bit from the book, but it's equivalent. 
<a href="#ref61">↩</a>
</div>

<p>
</p>

<div id="foot62"> 62. "Additive" because we are summing the $f_i$. "Generalized" because it generalizes from the linear functions $\beta_jX_j$ in ordinary linear regression.
<a href="#ref62">↩</a>
</div>

<p>
</p>

<div id="foot63"> 63. It's not hard to see that (with the exception of local regression), all the models discussed in this chapter can be seen as special cases of GAM.
    
<a href="#ref63">↩</a>
</div>
