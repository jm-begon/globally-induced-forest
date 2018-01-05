# Globally Induced Forest

Globally Induced Forest is a python package to build lightweight yet accurate decision forests. The core tree implementation is based on scikit-learn 0.18. The provided estimator are [scikit-learn](http://scikit-learn.org/stable/) compatible.

If you use this package, please cite

    Begon, J. M., Joly, A., & Geurts, P. (2017, July). Globally Induced Forest: A Prepruning Compression Scheme.
    In International Conference on Machine Learning (pp. 420-428).

The paper is avaiblable at http://orbi.ulg.ac.be/handle/2268/214279.

# Dependencies

The framework was tested under python 2.7 and 3.5 with the following dependencies:

 * six 1.10.0
 * numpy 1.11.0
 * scipy 0.17.1
 * scikit-learn 0.18.1

 You may use `pip install -r requirements.txt` to install all requirements.

# Install

To install this package, clone the repository locally and use

    python setup.py install

# Examples

The examples folder contains a couple of snipets demonstrating the use of `GIFClassifier` and `GIFRegressor`.
