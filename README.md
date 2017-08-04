# Linear Regression Algorithms
This project implements the following algorithms for linear regression:
1. Closed-Form (Mathematical) Solution
2. Gradient Descent
## 1. Closed-Form (Mathematical) Solution
This solution implemented in `ClosedForm_Regression.py` implements the following formula:
* W = (H<sup>T</sup>H)<sup>-1</sup>H<sup>T</sup>y

Where:
* y is a `|N|` column vector of labels,
* H is a `|NxD|` Matrix of N observations with D features each
* and W is a `|D|` column vector the obtained Weights.
## 2. Gradient Descent
This solution implements the renowned Gradient Descent Algorithm in `GradientDescent_Regression.py`

License
-------

The MIT License. Copyright (c) 2017 Amin Fadaee

About Author
----------------

[Amin Fadaee](https://www.linkedin.com/in/aminfadaee/)
