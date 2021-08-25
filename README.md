# py_ml_2021

## About this Repo

These are a series of machine learning models implemented in Python without
any ML libraries. I've used an API interface similar to TensorFlow with some small
changes, so anyone who's used TensorFlow will probably find this reasonably intuitive.

This is a synthesis of everything I've learned about ML which I'm using primarily
as a portfolio piece to demonstrate an understanding of translating theory to code.

Hopefully you can either gain new understanding of these models, or feel
more comfortable working with me!

## Usage

/models contains the base Model class along with each individual ML model.

Each model has an accompanying test file in the main directory. You can run these to see a plot of the costs. It will also print a classification rate.

The models are all trained on the MNIST dataset, so you'll need to create a data folder
with train.csv inside to run this code.

## Acknowledgements

I've taken many courses from an instructor with the username [LazyProgrammer](https://www.udemy.com/user/lazy-programmer/). I owe much of my understanding of the theory
behind machine learning to him. If you're interested in learning more about ML, definitely check out his courses.