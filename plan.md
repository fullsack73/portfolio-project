# Objective

1. Find a way to automate the portfolio optimization process
- user inputs desired risk-free asset rate, risk tolerance, and target return(and date range). this might accomodate options like risk rate "below" certain value, target return "above" certain value, etc.
- the program finds the portfolio(s) that match user's preferences
- could be acheived by:
    - doing brute force search over all possible portfolio combinations for limited number of tickers(e.g. from S&P 500 or whatever user prefers)
    - using some kind of optimization algorithm that does not requires brute forcing, i doubt this is even possilbe though.
    - using machine learning. but this still requires data from all possible portfolio combinations, so it's not much of an improvement.
