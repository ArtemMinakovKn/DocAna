# DocAna
Project for DocAna

***Authots:*** *Artem Minakov, Linus Krause, Elena Putilova, Diana Sharafeeva*

## Project goals 

Top 20 countries from the 2023 World Happiness Report 

[Link to the research](https://worldpopulationreview.com/country-rankings/happiest-countries-in-the-world)

![map](link)

## Data 
For the projects used Rediit data from datasets library in python

```python
from datasets import load_dataset
dataset = load_dataset("reddit")
```
After downloading the data the following countries subreddits (according to the "Happiness list') were choosen for futher analysis: 
 - Finland
 - Denmark
 - Switzerland
 - Iceland
 - Netherlands
 - Norway
 - Sweden
 - Luxembourg
 - Newzealand
 - Austria
 - Australia
 - Israel
 - Germany
 - Canada
 - Ireland
 - Costarica
 - UK 
 - Czechrepublic
 - Belgium
 - France
 - Malta

Several countries from top-20 list were skipped because of the lack of the apptopriate amount of text in subredits (less than 150 texts).

## Model evaluation 

To evaluate the performance of the model with different languages
BERT sentiment analyser on French original vs French translated to English subreddit to compare how the translation and the BERT on original and translated data works. 

Outcome: BERT works well both ways => 34% are not the same, but the analyzer still works quite good (discrepancy in data, where it exists, is mostly 1)


Project Progress:
Extracting Reddit posts from respective countries.
Translating Reddit posts into English using Google API.
Conducting sentiment analysis on original and translated Reddit posts using BERT.
Comparing sentiment analysis results from BERT and VADER to determine the best method.
Calculating the average sentiment for each country.


## Sentiment analysis 
Performing sentimental analysis on reddits for each country 

## Topic modelling 

## Outcome

## Challenges 

- Insufficient amount of data for certain countries
Language translation:
google API is not the best translator while really good ones are either non-free or limited in terms of tokens (512 tokens). 


Sentiment: limited number of tokens for BERT sentiment: text was divided into chunks of text
LDA topic modeling: 
BERTopic: 
