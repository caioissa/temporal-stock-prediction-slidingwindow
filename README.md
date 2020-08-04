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
python src/get_data.py
```
## Usage
1. Train the NN and save the model
```bash
python src/train.py
```
2. Test the model in a simulation for 50 unseen consecutive days (the parameter refers to the initial wallet ammount)
```bash
python src/simulation.py 1000
```
