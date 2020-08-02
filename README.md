# Stockmarket Predictions
### Deep learning and sliding windowing for temporal series of US Stockmarket data
* Sliding window of historic stockmarket data was used to train a feedforward neural network.
* The validation with a wallet simmulation showed a profit of 200% for 50 days of unseen data.
## Installation
Create a virtualenv and run
```bash
pip install -r requirements.txt
```
## Extract Data
Get the data for RUN stock (Sunrun Inc.) for the last year (from today)
```bash
python get_data.py
```
## Usage
1. Run the stockmarket.py script to train the NN and save the model
```bash
python stockmarket.py
```
2. Run the stockclient.py script for validation (the parameter refers to the initial wallet ammount)
```bash
python stockclient.py 1000
```
