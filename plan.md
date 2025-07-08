# Objective

1. Find a way to automate the portfolio optimization process
- user inputs desired risk-free asset rate, risk tolerance, and target return(and date range). this might accomodate options like risk rate "below" certain value, target return "above" certain value, etc.
- the program finds the portfolio(s) that match user's preferences
- could be acheived by:
    - doing brute force search over all possible portfolio combinations for limited number of tickers(e.g. from S&P 500 or whatever user prefers)
    - using some kind of optimization algorithm that does not requires brute forcing, i doubt this is even possilbe though.
    - using machine learning. more on this below.

## Problem with Machine Learning
- i don't know what user will input as date range, so i have no choice but to use fixed date range to train the model. (e.g. 1 year, 3 years, etc.)
- i need to find a way to train the model and include it in the program, so it does not have to generate model everytime.
- the tickers for the machine learning also have to be fixed, and i have to figure out which group of tickers would be adequate for general use.(or i could make several models for several different groups. e.g. one with S&P 500, one with Dow Jones, and each model for each sectors, etc. though this might make the program way heavier than it already is.)