from deap.tools import mutation
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Download historical stock data
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-12-31'
data = yf.download(symbol, start=start_date, end=end_date)

# Define the fitness function
def fitness_function(chromosome, stock_data):
    short_period, long_period = chromosome

    # Calculate the moving averages
    stock_data['SMA_short'] = stock_data['Close'].rolling(window=short_period).mean()
    stock_data['SMA_long'] = stock_data['Close'].rolling(window=long_period).mean()

    # Generate trading signals
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['SMA_short'] > stock_data['SMA_long'], 'Signal'] = 1
    stock_data['Signal'] = stock_data['Signal'].diff()

    # Implement the trading strategy
    stock_data['Position'] = stock_data['Signal'].cumsum().clip(lower=0, upper=1)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['Strategy_Returns'] = stock_data['Position'].shift(1) * stock_data['Returns']

    # Calculate the strategy performance
    total_return = stock_data['Strategy_Returns'].sum()
    annualized_return = (1 + total_return) ** (252 / len(stock_data)) - 1

    return annualized_return,

# Mutation function
def mutation(chromosome, mutation_rate, min_period, max_period):
    mutated_chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = random.randint(min_period, max_period)
    return tuple(mutated_chromosome),

# Create the types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize the DEAP components
toolbox = base.Toolbox()
toolbox.register("attr_short", random.randint, 5, 50)
toolbox.register("attr_long", random.randint, 51, 200)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_short, toolbox.attr_long), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutation, mutation_rate=0.1, min_period=5, max_period=200)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function, stock_data=data)

# Initialize the population
pop = toolbox.population(n=100)

# Set the algorithm parameters
ngen = 100  # number of generations
cxpb = 0.8  # crossover probability
mutpb = 0.2  # mutation probability

# Store the fitness values during the optimization process
avg_fitness_values = []
best_fitness_values = []

# Define a function to store the fitness values at each generation
def store_fitness_values(population):
    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length
    best = max(fits)

    avg_fitness_values.append(mean)
    best_fitness_values.append(best)

# Run the genetic algorithm
for gen in range(ngen):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    offspring = list(offspring)

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace the old population by the offspring
    pop[:] = offspring

    # Store the fitness values at the current generation
    store_fitness_values(pop)

# Extract the best individual (trading strategy) from the final population
best_individual = tools.selBest(pop, k=1)[0]
best_short_period, best_long_period = best_individual

# Calculate and display the performance of the optimized strategy
# using the best short and long moving average periods
optimized_annualized_return = fitness_function(best_individual, data)[0]
print(f"Best strategy: Short period = {best_short_period}, Long period = {best_long_period}")
print(f"Optimized annualized return: {optimized_annualized_return:.4f}")

# Visualize the evolution of the population's fitness over time
plt.figure(figsize=(10, 5))
plt.plot(avg_fitness_values, label='Average Fitness')
plt.plot(best_fitness_values, label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc='upper left')
plt.title('Evolution of Fitness over Generations')
plt.show()

# Plot the stock prices and moving averages for the optimized strategy
data['Optimized_SMA_short'] = data['Close'].rolling(window=best_short_period).mean()
data['Optimized_SMA_long'] = data['Close'].rolling(window=best_long_period).mean()

plt.figure(figsize=(15, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Optimized_SMA_short'], label=f'Optimized SMA Short ({best_short_period} days)')
plt.plot(data['Optimized_SMA_long'], label=f'Optimized SMA Long ({best_long_period} days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.title('Stock Prices and Optimized Moving Averages')
plt.show()
