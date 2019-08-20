---
layout: page
title: 10. Unsupervised Learning
---

{% katexmm %}

# Exercise 7: Comparison of correlation based distance and Euclidean distance on `USArrests` dataset.

For this exercise, we'll just show the proportionality holds in general.

The authors mention (p397) that "this is an unusual use of correlation, which is normally computed between variables; here it is computed between observation profiles". It [appears](https://www.datanovia.com/en/lessons/clustering-distance-measures/) the authors intended that for observations $x_i, x_j \in \mathbb{R}^p$, 

$$
\begin{aligned}
r_{ij} &= \frac{\sum_{k = 1}^p (x_{ik} - \overline{x}_i)(x_{jk} - \overline{x}_j)}{\sqrt{\sum_{k = 1}^p(x_{ik} - \overline{x}_i)^2 \sum_{k = 1}^p(x_{jk} - \overline{x}_j)^2}}\\
&= \frac{(x_i - \overline{x}_i)^\top (x_j - \overline{x}_j)}{||x_i - \overline{x}_i||^2||x_j - \overline{x}_j||^2}
\end{aligned}
$$

where $\overline{x}_i = \frac{1}{p}\sum_{k = 1}^p x_{ik}$ is the mean over the features. This can be seen the correlation of the pairs $(x_{ik}, x_{jk})$, $k = 1, \dots, p$, (hence the use of the word "unusual" - the feature index has become a sample index).

If the data has been standardized then 

$$\overline{x}_i = \overline{x}_j = 0$$

and 

$$||x_i - \overline{x}_i||^2 = ||x_j - \overline{x}_j||^2 = 1$$

so the correlation

$$r_{ij} = \sum_{k = 1}^p x_{ik}x_{jk} = x_i^\top x_j $$

and the squared euclidean distance is

$$
\begin{aligned}
||x_i - x_j||^2 &= x_i^\top x_i - 2x_i^\top x_j + x_j^\top x_j \\
&= 2(1 - x_i^\top x_j)\\
&= 2(1 - r_{ij})
\end{aligned}
$$

Hence

$$1 - r_{ij} \propto ||x_i - x_j||^2$$

as claimed.

{% endkatexmm %}