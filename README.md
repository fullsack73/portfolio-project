# Portfolio Stuff Program

## Overview
this program shows stock data, regresion line, hedge info, and provides optimized portfolio

## ⚠️ **Disclaimer**
**This program is only meant for academic purposes and should not be taken as a serious financial analysis.**  
**Use the result you get from the program AT YOUR OWN RISK.**

## Features

### 1. Stock Data Visualization
- shows financial timeline for given ticker

### 2. LightGBM Regression
- shows regression data(prediction of the price)
- uses LightGBM to predict the price

### 3. Hedge Relation Analysis
- it tells you the 2 stocks you provided are in hedge or not

### 4. Portfolio Optimization
- tells you how to split your money (because apparently, you need help with that)
- uses basic math to find the "optimal" portfolio weights (spoiler: past performance doesn't guarantee future results, but we pretend it does)
- shows you the efficient frontier (it's like a buffet line for investments, but with less food and more risk) 
- shout out to Harry Markowitz, an absolute legend

### 5. Acknowledgements
- thanks to yfinance for letting me steal their data (legally, of course)
- pandas and numpy for doing all the heavy lifting while I just write pretty code