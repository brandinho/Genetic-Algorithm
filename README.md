# Genetic-Algorithm
Implementation of Genetic Algorithms for both binary and continuous variable optimization (WORK IN PROGRESS)

The binary implementation was designed for feature selection. We use an artificial neural network as the example algorithm, but any algorithm can be used. Assuming we encode binary vectors as such: [0,1,1,0,1,1,1,0,0,1,0], with 0 and 1 meaning that we discard and keep the variables respectively, we can run the genetic algorithm to determine the optimal feature mix for our model.

The continuous implementation was designed to optimize neural network weights in a Reinforcement Learning setting. Slight modifications are done to the mutation and generation functions to deal with continuous variables.
