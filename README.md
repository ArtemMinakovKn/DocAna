# Are the happiness polls saying the truth?

***Authots:*** *Artem Minakov, Linus Krause, Elena Putilova, Diana Sharafeeva*
*Team: Konstanz Lovers*

## Project goals 

The primary objective of the project was to investigate whether the countries identified as the "happiest in the world" also exhibited a higher level of positive sentiment within their respective Reddit communities. Specifically, the focus was placed on analyzing the subreddits associated with the 20 top-ranked countries as identified in the [2023 World Happiness Report](https://worldpopulationreview.com/country-rankings/happiest-countries-in-the-world). In the project applied the techniques learned during the course "Document Analysis with Computational Methods" at Konstanz Universitat.

![map](https://github.com/ArtemMinakovKn/DocAna/blob/main/happiness_countries.jpg)

The project involved the following steps:
- Subreddit Extraction: The subreddits corresponding to the 20 countries listed in the happiness report were extracted. Subreddits with a minimum of 100 Reddit posts were retained for analysis.  
- Comparison of Sentiment Analysis Models: A comparison was made between manual sentiment analysis, the VADER model, and the BERT model using the New Zealand subreddit as a case study. The objective was to determine which model performed better in terms of sentiment analysis.
- Comparison of perfomance BERT Sentiment model on original and translated text: The BERT sentiment analyzer was applied to both original French Reddit posts and their translated English versions. 
- Translation of Non-English Reddits: Google Translator API was employed to translate Reddit posts written in languages other than English into English for further analysis.
- BERT Sentiment Analysis on Translated Data: BERT sentiment analysis was conducted on the translated Reddit data, encompassing all languages, to assess the overall sentiment across the selected countries.
- Comparison of Sentiment Ranking: The sentiment analysis results were compared with the initial research paper's happiness scores to determine the similarity in country rankings based on sentiment analysis and the original happiness scores.
- Topic Modeling: To explore potential explanations for variations in sentiment scores, topic modeling was performed to identify the predominant topics discussed within each country's subreddit.

## Data 
For the projects used Rediit data from datasets library in python

```python
from datasets import load_dataset
dataset = load_dataset("reddit")
```
After downloading the data the following countries subreddits (according to the "Happiness list') were chosen for further analysis:
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

Several countries from the top-20 list were skipped because of the lack of the appropriate amount of text in subreddits (less than 150 reddit posts in Luxembourg, Netherlands, Malta, Costa Rica) and Iceland was excluded later because of the impossibility to properly translate the texts into English.

The chart below represent the 30 most frequently used words across all country Reddits.

![map](https://github.com/ArtemMinakovKn/DocAna/blob/main/30freq_words.jpg)

## Comparison of manual sentiment vs VADER vs BERT
To determine the best sentiment analysis library, it was decided to manually examine a certain number of Reddit posts (around 50), where two models showed completely opposite results. Overall, the advantage of BERT is undeniable. Here's an example of a Reddit post where VADER gave a highly positive rating while BERT gave a highly negative rating:
“I got up on the first day of the school holidays to find my kitchen floor coated in sugar, uneaten cornflakes on the bench (because they cocked up and put salt on them first) and my 5 and 7 year olds drinking straight sugar from their bunny cups. Thank God I could flick them to their grandmother yesterday while I worked, I may have throttled the pair of them.”
The choice of the library becomes evident.

## Model evaluation 

Given the diversity of languages used in the selected countries' Reddit communities, it was necessary to assess the performance of the sentiment analysis models on both original and translated Reddit posts. This evaluation aimed to ensure that the models are capable of effectively handling sentiment analysis across different languages in translated texts as well as in original. 
The results indicated that the BERT model performed effectively in both scenarios. Although there was a slight discrepancy of approximately 34% in the resulting sentiment scores, with the majority of discrepancies amounting to only 1, the overall performance of the analyzer remained satisfactory. These findings suggested that the BERT model could successfully handle sentiment analysis on both original and translated data.

## Text translation

 To apply BERT model for different subredits languages the Google API translator was used. 

The Google API translator facilitated the automatic translation of Reddit posts, streamlining the process. Additionally, it proved to be cost-effective compared to alternative translation methods. 
However, this approach did have certain drawbacks. Firstly, based on previous experience with Google Translate and other alternative resources, it was acknowledged that Google Translate may not be the most efficient option available. Moreover, there is a character limit of 5,000 characters per translation request. To address this limitation, the Reddit posts were split into multiple batches. However, it should be noted that this splitting process may have had an impact on the quality of the translations.

Although there are more successful online translators, they offer limited functionality within their free versions. Therefore, the decision was made to proceed with the Google API translator despite its drawbacks, considering the overall efficiency and cost-effectiveness it provided for the project's requirements.

## Sentiment analysis 
The sentiment analysis of Reddit posts from different countries was performed using the bert-base-multilingual-uncased model 'nlptown/bert-base-multilingual-uncased-sentiment'. 
This model has been specifically fine-tuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish, and Italian. It assigns a sentiment score to each review, ranging from 1 to 5 stars.
Due to the model's token limitation of 512, a text processing methodology described in a relevant [article](https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f) was adopted. Specifically, the text of each Reddit post was segmented into chunks of 512 tokens. These segments were subsequently transformed into individual tensors and incorporated into the input dictionary for the BERT model. The final sentiment score, representing the mean value of all text segments, was then derived from the model's predictions.
To enable cross-country sentiment comparison, the overall sentiment score for each country was calculated as the average sentiment value across all Reddit posts originating from that specific country. This calculation provided a comprehensive assessment of sentiment variation among the countries under consideration.

![map](https://github.com/ArtemMinakovKn/DocAna/blob/main/sent_score_countries.jpg)

## Topic modelling 
As the next step of the analysis we wanted to apply topic modeling to subreddits of different countries and find out what are the most popular themes for discussion in each country. Ideally, the plan was to extract the meaningful topic labels and the keywords that would make it possible to explain the difference in how positive or negative the reddits from the countries are. To do so, we used two approaches: LDA (as the most widely used and simple one) and BERTopic.

Topic modeling is a statistical technique used to uncover underlying themes or topics within a collection of documents. It helps in identifying the main ideas or subjects that appear across the text data by analyzing patterns of word usage. 
Latent Dirichlet Allocation (LDA): LDA is a generative probabilistic model that assumes each document is a mixture of multiple topics, and each topic is a distribution of words. It aims to discover the latent topics that best explain the observed document-word patterns.

The results of the conducted LDA topic modeling can be found below. In the picture with the stacked bar plot we can see the distribution of the 10 topics across all the 15 countries. However, the results of the topic modeling itself could not be meaningfully interpreted by the provided keywords. We found out that LDA gives bigger weights to proper nouns when defining the topics and some topics overlap by these nouns. The similarity of the topics was examined by the similarity matrix and can be found as a heat map plot below.

```python
[(0,
  '0.003*"harper" + 0.002*"telstra" + 0.002*"sydney" + 0.002*"canadians" + 0.002*"ontario"'),
 (1,
  '0.007*"dublin" + 0.005*"nhs" + 0.003*"ukip" + 0.002*"cork" + 0.002*"scottish"'),
 (2,
  '0.004*"bell" + 0.004*"harper" + 0.003*"quebec" + 0.003*"canadians" + 0.002*"swedes"'),
 (3,
  '0.007*"nz" + 0.003*"canadians" + 0.003*"nhs" + 0.003*"harper" + 0.003*"quebec"'),
 (4,
  '0.008*"nz" + 0.005*"harper" + 0.005*"canadians" + 0.005*"quebec" + 0.004*"bell"'),
 (5,
  '0.011*"palestinian" + 0.008*"israelis" + 0.006*"gaza" + 0.005*"palestine" + 0.004*"settlements"'),
 (6,
  '0.006*"canadians" + 0.005*"quebec" + 0.005*"bell" + 0.004*"harper" + 0.003*"ontario"'),
 (7,
  '0.016*"nz" + 0.007*"auckland" + 0.005*"nbn" + 0.004*"telstra" + 0.004*"maori"'),
 (8,
  '0.004*"bell" + 0.004*"nbn" + 0.003*"harper" + 0.003*"canadians" + 0.003*"australians"'),
 (9,
  '0.007*"canadians" + 0.006*"harper" + 0.005*"quebec" + 0.005*"ontario" + 0.005*"bell"')]
  ```
![map](https://github.com/ArtemMinakovKn/DocAna/blob/main/LDA_topics.jpg)

![map](https://github.com/ArtemMinakovKn/DocAna/blob/main/LDA_heatmap.jpg)

Since the LDA method did not lead us to the results we expected, we decided to continue our analysis further and moved to the BERTopic module.

In the BERTopic framework we used Sentence-BERT (SBERT) to create document embeddings. We assume that documents containing the same topic are semantically similar. These embeddings are primarily used to cluster semantically similar documents and not directly used in generating the topics, this is done later on.

- Dimension Reduction

We reduced embeddings using UMAP. UMAP has some advantages over e.g. PCA.
Non-linear Dimension Reduction:
UMAP is a non-linear dimension reduction technique, while PAC is a linear method. Topic modeling often deals with complex, non-linear relationships among words and topics. UMAP is capable of capturing and preserving nonlinear structures in the data.

 Preserving Local Structure:
UMAP's algorithmic approach focuses on preserving these local relationships, resulting in more meaningful and interpretable representations. PAC, on the other hand, primarily focuses on preserving global variances in the data, which may not fully capture the fine-grained relationships between words.

Scalability and Efficiency:
UMAP is known for its scalability and computational efficiency, making it suitable for processing large text datasets.

We decided for the following parameters: n_neighbors=15, n_components=5. The common default value for number neighbors is 15. A relatively large value for the number of neighbors gives more of the big picture and less of the details. That's what we were looking for to limit the amount of topics generated and not get single reddit topics.

- Clustering

The reduced embeddings are clustered using HDBSCAN. It is an extension of DBSCAN that finds clusters of varying densities by converting DBSCAN into a hierarchical clustering algorithm. In high dimensional data distance clustering algorithms are disadvantageous. We decided on a minimum cluster size of 20. The idea was to not allow single reddits influence/create topics by their own and create more general bigger topics. One could adjust these parameters but needs to be aware that the topics might be very specific and not representative.

- Weighing

A modified TF-IDF score is calculated, such that it allows for a representation of a term’s importance to a topic instead. All documents in a cluster are treated as a single document by simply concatenating the documents. Then a c-TF-IDF score is calculated.

- Fine Tuning

For fine tuning we applied stop word removal to create more meaningful topics. We do this after creating embeddings because the transformer-based embedding models that we use need the full context in order to create accurate embeddings.

- Top 10 Topics
The amount of reddits that were available influenced the diversity of topics. Less reddits lead to less extracted topics and therefore the remaining topics account for more of the total.


- Topic keywords
The length of the bar represents the score of the keywords, the most "important"/representative words (c-tf-idf score). A longer bar means higher importance for the topic.

![map](https://github.com/ArtemMinakovKn/DocAna/blob/main/topics_sweden.png)


- Resume BERTopic

One could further limit the words used for clustering by only using NE and nouns or by doing POS-Tagging using spacy. We tried it and decided to leave the topics in their raw form. Also removing all locations doesnt seem like the way to go. As one can observe for the topics in sweden, there's one cluster containing keywords like russia, nato & china. Removing those locations would be problematic for the cluster.  
We encourage others to try out other approaches to yield more fine grained topics. Also we compared topics between countries manually due to the number of topics that were produced. For a bigger amount of topics this could be done using BERTopic.
Overall we got some interesting insights into some topics for countries with a larger amount of reddits (e.g. canada) but unfortunately no supporting insights regarding sentiment analysis.

## Outcome

The findings reveal that Finland, Norway, and Germany exhibit the highest proportion of positive sentiment among the analyzed Reddit posts. Conversely, Israel, Canada, and Australia display the lowest share of positive sentiment. These variations in sentiment can be attributed to the specific topics or issues discussed by Reddit users in each country. The prevalence of certain problem-oriented topics appears to influence the overall sentiment scores observed.  
Furthermore, it is important to acknowledge that the performance of the employed sentiment analysis model, 'nlptown/bert-base-multilingual-uncased-sentiment', may vary across different languages. This discrepancy stems from the inherent differences in nuances, grammar, and sentiment expressions across languages, which can impede the model's ability to accurately capture and interpret sentiment uniformly across all languages.  
Comparing these results with the Happiest countries research findings, it becomes evident that Finland and other North European countries consistently occupy top positions in both ratings. This observation suggests that these countries tend to exhibit higher overall sentiment scores compared to other countries under analysis.

![map](https://github.com/ArtemMinakovKn/DocAna/blob/main/sent_score.jpg)
