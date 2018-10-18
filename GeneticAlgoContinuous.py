# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:38:03 2018

@author: dasilvb
"""
import tensorflow as tf
import numpy as np



# Perform crossover from two selected parents
def crossover(parent_1, parent_2, crossover_probability, population_size, crossover_type, random_type = False):
    crossover_flag = np.random.random() < crossover_probability
    # We can choose to randomly select which crossover type to use
    # If this is set to False, then we have to select one of the options in our function inputs
    if random_type:
        crossover_type = np.random.choice(np.array["One Point", "Two Point", "Uniform", "Arithmetic"])
    which_parent = np.random.choice(np.array(["Parent 1", "Parent 2"]))
    if crossover_flag == True:
        if crossover_type == "One Point":
            # One Point crossover randomly selects a partition point in the chromosome vector and takes the head and tail of 
            # parent_1 and parent_2 respectively to for the new chromosome         
            # We are making the assumption that the parent_1 vector is one dimensional
            partition = np.random.randint(0, parent_1.shape[0])
            if which_parent == "Parent 1":
                child = parent_1
                child[partition:] = parent_2[partition:]
            elif which_parent == "Parent 2":
                child = parent_2
                child[partition:] = parent_1[partition:]
        elif crossover_type == "Two Point":
            # Two Point crossover is similar to one point, except now we have to reference points and it is the portion 
            # in between that gets swapped
            lower_limit = np.random.randint(0, parent_1.shape[0]-1)
            upper_limit = np.random.randint(lower_limit+1, parent_1.shape[0])
            if which_parent == "Parent 1":
                child = parent_1
                child[lower_limit:upper_limit+1] = parent_2[lower_limit:upper_limit+1]
            elif which_parent == "Parent 2":
                child = parent_2
                child[lower_limit:upper_limit+1] = parent_1[lower_limit:upper_limit+1]    
        elif crossover_type == "Uniform":
            # Uniform randomly selects indexes with a uniform distribution to be swapped during crossover
            # Unlike the previous two methods, the genes to be swapped do not have to be in a sequence
            random_sequence = np.random.choice(np.arange(parent_1.shape[0]), np.random.randint(1, parent_1.shape[0]), replace = False)
            if which_parent == "Parent 1":
                child = parent_1
                child[np.sort(random_sequence)] = parent_2[np.sort(random_sequence)]
            elif which_parent == "Parent 2":
                child = parent_2
                child[np.sort(random_sequence)] = parent_1[np.sort(random_sequence)]
        elif crossover_type == "Arithmetic":
            # Rather than swap genes to form a new chromosome, we are doing some arithmetics to make a new chromosome
            random_weight = np.random.rand()
            child = parent_1 * random_weight + parent_2 * (1 - random_weight)
        else:
            raise ValueError("Sorry, that is not one of the options")        
    else:
        # There is probability (1 - p) that we will not perform crossover
        # In such an event the child takes the gene of one of the parents (randomly selected)
        if which_parent == "Parent 1":
            child = parent_1
        elif which_parent == "Parent 2":
            child = parent_2
    return child



# Mutate a portion of the gene
def mutation(population, mutate_probability, population_size, noise_scale):
    # We start at position 2 because we don't want to mutate the two parents that we are carrying forward
    # This is called elitism, and we do it to always preserve the best chromosomes through the generations
    for p in range(2, population_size):            
        child = population[p,]
        mutate_flag = np.random.random() < mutate_probability
        if mutate_flag == True:
            # noise_scale determines how much noise we want to use in the mutation (the variance of the distribution)
            noise = np.random.standard_normal() * noise_scale
            mutation_position = np.random.randint(0, population.shape[1])
            child[mutation_position] = child[mutation_position] + noise
            if child[1] < child[0]:
                child[1] = child[0] + 1
        population[p,] = child
    return population



# Generate a new population with gaussian noise
def generate_gaussian_population(population, competition_scores, elite_population, selection_method, 
                                 population_size, feature_size):
    new_population = np.zeros((population_size, feature_size))
    # We are using the concept of elitism, so we keep the top two performers from the previous generation
    new_population[:2,] = elite_population
    # We also want to make sure to use them each at least once in the "breeding" process
    new_population[2:4,] = elite_population
        
    if any(np.array(competition_scores) < 0): competition_scores += -min(competition_scores)
    population_index_sorted = np.argsort(competition_scores)[::-1]
    competition_scores_sorted = np.sort(competition_scores)[::-1]
    scores_cumulsum = np.cumsum(competition_scores_sorted)
    for p in range(2, population_size):
        if p > 3:
            if selection_method == "Roulette":
                random_number = np.random.uniform(low = 0, high = scores_cumulsum[-1])
                if all(scores_cumulsum == scores_cumulsum[0]):
                    position = 0
                else:
                    # Randomly select the parent with a probability proportional to their competition score
                    position = next(x[0] for x in enumerate(scores_cumulsum) if x[1] > random_number)
                parent = population[population_index_sorted[position],]
            elif selection_method == "Tournament":
                # Host a mini tournament to see who will become the parent that we add gaussian noise to
                k = population_size // 2
                tournament_population = np.zeros((k, 1))
                total_competitors = np.random.choice(np.arange(population_size), k, replace = False)
                tournament_population[:,0] = competition_scores[total_competitors]
                
                parent_index = total_competitors[np.argmax(tournament_population, axis = 0)]
                parent = population[parent_index,]
        else:
            parent = population[p]
        
        new_population[p,] = parent + np.random.standard_normal(parent.shape[0])

    return new_population



# Generate a new population with crossover and mutation
def generate_population(population, competition_scores, elite_population, selection_method, crossover_probability, 
                        crossover_type, random_type, population_size, feature_size):
    new_population = np.zeros((population_size, feature_size))
    # We are using the concept of elitism, so we keep the top two performers from the previous generation
    new_population[:2,] = elite_population
    
    if any(np.array(competition_scores) < 0): competition_scores += -min(competition_scores)
    population_index_sorted = np.argsort(competition_scores)[::-1]
    competition_scores_sorted = np.sort(competition_scores)[::-1]
    scores_cumulsum = np.cumsum(competition_scores_sorted)
    for p in range(2, population_size):
        if selection_method == "Roulette":
            position = []
            for i in range(2):
                random_number = np.random.uniform(low = 0, high = scores_cumulsum[-1])
                if all(scores_cumulsum == scores_cumulsum[0]):
                    position.append(0)
                else:
                    # Randomly select the parents with a probability proportional to their competition score
                    position.append(next(x[0] for x in enumerate(scores_cumulsum) if x[1] > random_number))
            parent_1 = population[population_index_sorted[position[0]]]
            parent_2 = population[population_index_sorted[position[1]]]
        elif selection_method == "Tournament":
            # Host two mini tournaments to see who will become the parents for crossover
            k = population_size // 2
            tournament_population = np.zeros((k, 2))
            total_competitors = np.random.choice(np.arange(population_size), k * 2, replace = False)
            tournament_population[:,0] = competition_scores[total_competitors[:k]]
            tournament_population[:,1] = competition_scores[total_competitors[k:]]
            
            parent_indexes = total_competitors[np.argmax(tournament_population, axis = 0) + np.array([0,k])]
            parent_1 = population[parent_indexes[0],]
            parent_2 = population[parent_indexes[1],]
        
        new_population[p,] = crossover(parent_1, parent_2, crossover_probability, population_size, crossover_type, random_type)

    return new_population



# Run the genetic algorithm
def genetic_algorithm(run_competition_function, neural_network_shell, num_generations, population_size, gaussian_or_crossmutate, 
                      crossover_probability, crossover_type, random_type, mutate_probability, noise_scale, selection_method, 
                      neurons, *args, **kwargs):
    # This array will keep track of the best scores for each generation
    best_scores = np.zeros(num_generations)

    # The specific example I'm showing below is for a reinforcement learning agent that is a parameterized neural network
    tf.reset_default_graph()
    sess = tf.Session()
    agent = neural_network_shell(sess, neurons, population_size)
    sess.run(tf.global_variables_initializer())
    
    for gen in range(num_generations):

        # You can define whatever competition function that you like and replace the one below
        # You can also put in actual function arguments instead of *args and **kwargs
        # We are assuming you have initialized your agent class with a population size attribute
        competition_scores = np.array([run_competition_function(*args, **kwargs) for agent_index in range(agent.population_size)])

        best_scores[gen] = np.max(competition_scores)
        # We are assuming you initialized your agent class with a numpy array (population attribute) 
        # to store chromosomes (i.e. neural network weights)
        elite_population = agent.population[competition_scores.argsort()[-2:][::-1],]
        if gaussian_or_crossmutate == "Gaussian":
            agent.population = generate_gaussian_population(agent.population, competition_scores, elite_population, selection_method, population_size, agent.population.shape[1])
        elif gaussian_or_crossmutate == "Crossmutate":
            agent.population = generate_population(agent.population, competition_scores, elite_population, selection_method, 
                                                   crossover_probability, crossover_type, random_type, population_size, agent.population.shape[1])
            agent.population = mutation(agent.population, mutate_probability, population_size, noise_scale)

