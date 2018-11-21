# BPR-by-tensorflow

implement bayesian personalized ranking by tensorflow

-----------------------------------------------------------------------------

## Introduction
bayesian personalized ranking is a popular method in recommender system .
In recent year, a lot of recommender system study combine deep learning model with BPR training loss.

## dataset
The dataset is about the the user's rating to the moives. This is from Kaggle:(https://www.kaggle.com/rounakbanik/the-movies-dataset).

## usage

Training tuple:
	<userID , positive itemID , negative itemID>
   
To train a model:
	
    $ python main.py
    
## Acknowledgement
I implement this method [BPR: Bayesian Personalized Ranking from Implicit Feedback. Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme, Proc. UAI 2009](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) by tensorflow
