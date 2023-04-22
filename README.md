# Stock Trading Optimization with Genetic Algorithm

This project demonstrates the application of a genetic algorithm to optimize a moving average crossover trading strategy for a given stock. The goal is to maximize the annualized return of the trading strategy by finding the optimal short and long moving average periods.

## Overview

The project consists of the following steps:

1. Download historical stock data using the `yfinance` library.
2. Define the fitness function, which calculates the annualized return of a trading strategy based on the input moving average periods.
3. Implement the genetic algorithm using the `DEAP` library.
4. Run the genetic algorithm to optimize the trading strategy.
5. Analyze and visualize the results.

## Requirements

- Python 3.7+
- yfinance
- pandas
- numpy
- matplotlib
- deap

## Usage
1. Download or clone this repository.
2. Open the Python file stock_trading_optimization.py and update the symbol, start_date, and end_date variables to the stock and time range you want to analyze.
3. Run the Python script using your preferred IDE or from the command line:
4. The script will download the historical stock data, run the genetic algorithm to optimize the moving average crossover strategy, and display the results, including the optimal short and long moving average periods and the annualized return of the optimized strategy.

## Results
The output of the script will display the best trading strategy found by the genetic algorithm, along with the annualized return of that strategy:

Best strategy: Short period = 42, Long period = 173

Optimized annualized return: 0.1118

Here is what it means - 

Best strategy: Short period = 42, Long period = 173: This indicates that the genetic algorithm has found the best moving average crossover trading strategy using a 42-day short moving average and a 173-day long moving average. In other words, the algorithm has determined that these periods for the short and long moving averages maximize the annualized return of the trading strategy based on the historical stock data.

Optimized annualized return: 0.1118: This value represents the annualized return of the optimized trading strategy using the 42-day short moving average and the 173-day long moving average. The annualized return is a measure of the average yearly return of the trading strategy over the historical period. In this case, the optimized trading strategy has an annualized return of approximately 11.18%.

In addition, the script will generate plots that visualize the fitness evolution over generations and the stock prices along with the optimized moving averages.

![Fitness Evolution](https://github.com/HarnoorSingh79/Stock-Trading-Optimization-with-Genetic-Algorithm/blob/main/Figure_1.png)

![Stock Prices and Optimized Moving Averages](https://github.com/HarnoorSingh79/Stock-Trading-Optimization-with-Genetic-Algorithm/blob/main/Figure_2.png)
