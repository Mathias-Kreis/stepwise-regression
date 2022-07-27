# stepwise-regression
Stepwise regression fits a logistic regression model in which the choice of predictive variables is carried out by an automatic forward stepwise procedure. The stepwise selection process can be visualized and shows the saturation in terms of AUC. Furthermore it is possible to use early stopping to get an parsimonious model.

![alt text](https://github.com/Mathias-Kreis/stepwise-regression/blob/main/stepwise_example.jpg?raw=true|width=50px)

## Installation Instructions
The package can be installed from GitHub using pip:
```python
pip install https://github.com/Mathias-Kreis/stepwise-regression/...
```

## Getting started
To use the functionality of stepwise-regression you just have to import the main function Logitstep as follows:
```python
from stepreg.stepreg import Logitstep
```

## Example
You can find an example in the main branch which models Fraud data that can be find on kaggle.

## Remark
The package is still under construction. I am thankful for hints and comments!
