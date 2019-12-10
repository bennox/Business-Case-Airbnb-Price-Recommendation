# RAMP starting kit on the AirBnB dataset
Source :  https://data.opendatasoft.com 

Authors: Léa Bresson, Flavien Gilles, Arthur Imbert, Eya Kalboussi, Benoît-Marie Robaglia (M2 Data Science)


[![Build Status](https://travis-ci.org/ramp-kits/boston_housing.svg?branch=master)](https://travis-ci.org/ramp-kits/boston_housing)
Go to [`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

Install ramp-workflow (rampwf), then execute

```
ramp_test_submission
```

to test the starting kit submission (`submissions/starting_kit`) and

```
ramp_test_submission --submission random_forest_100
```

to test `random_forest_100` or any other submission in `submissions`.

## Requirements

- numpy>=1.10.0
- matplotlib>=1.5.0
- pandas>=0.19.0
- scikit-learn>=0.17 (different syntaxes for v0.17 and v0.18)
- seaborn>=0.7.1
- nltk


## The project

This project aims at designing a competition problem based on a public dataset. We were also asked to provide a starting kit (notebook). This is what we propose :

AirBnB pricing is a ramp challenge that aims at predict the pricing of a listing on the AirBnB platform from various informations.

As Airbnb users are private individuals who are likely not aware of the real estate market, they (owners and lessee) can hardly decide if a price for a property is fair ; a too high price can be deterrent for the guests as a too low one will lose profit to both host and Airbnb.

The goal is to develop prediction models able to provide the "best price" for the owner taking into account the market environment and his house’s criterions to maximize both his benefits and the probability for his house to be rented. He is free to follow the recommendation or ignore it. Knowing how such an accomodation is usually priced on the platform can help future hosts at the time of choosing the fare of their listing (actually, hosts usually do the work of looking at similar offer to determine the price for their offer). The goal while helping them in this process is also to avoid too expensive or too cheap listings that would result in a poor experience of the platform and lower revenue both for the host and the platform. If a host decides to ignore the recommendation and sets lower price (to rent his house faster for instance), airbnb can inform the potential guests that the house is cheaper that the market price.

The data we will manipulate is from https://data.opendatasoft.com. The dataset gathers the listing of AirBnB in Paris area. It is divided in training and testing sets. The input contains of short texts describing the property (name, description, neighborhood_overview, neighbourhood, transit), plus some metadata. The output is the price.



