# Multi Layer Perceptron from scratch (using JAX)

This is an implementation from scratch of a multi layer perceptron made using [Google's JAX library](https://github.com/google/jax) for automatic differentiation and matrix operations.

It works reasonably well for both regression and classification problems.

Examples on how to use it can be seen at `classification_test.ipynb` and `regression_test.ipynb`.

The main upside of using JAX is that it makes computations pretty fast thanks to its JIT compiler, in fact I implemented the same model you can see in the classification test on Tensorflow and it took about 30 seconds less to achieve a very similar performance of 98% accuracy.

### Contribute

If you are willing to spend some time playing with matrices, fell free to contribute adding more features.

### Dependencies

- JAX
- Sklearn (only to shuffle data) using `sklearn.utils.shuffle`