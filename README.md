# [Kaggle competition summary: Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

This is a kaggle competition to classify the toxity of online forum comments.This competition lasts one month more than the typical two months as it changed the dataset once to get rid of the obvious data leakage found by many kagglers. I spent about three weeks working dilligently to achieve my final standing, **top 5% out of 4551 teams** which won me the second silver medal and places me to the **top 2% among more than 80,000 data scientists on Kaggle globally**.I'm hoping to win a gold medal some time year this to ascend to the honorary kaggle master title. Since I didn't team up with others to win this competition. I have a bit more to summarize than the Zillow one and the under-performed Porto Seguro one.

Like my previous two competitions, I again use ensemble-based meta-learning approach. This has become the standard thing to do if you want to win a kaggle competition. However, a two level stacking did not give me extra edge this time. Details about this are below.

 - Data cleaning / feature engineering

As I joined the competition only three weeks before the deadline. I didn't spent much time on feature engineering. Since this is competition is like a sentiment analysis which is basically natural language processing (NLP) problem, some NLP typical preprocessing were done. I used three word embeddings, Glove, FastText and Numberbatch. FastText appeared to perform the best. 

 - Single models

Models that go into my final stacking: 

Neural nets: GRU/LSTM, bidirectional GRU/LSTM, textCNN, Multi-column Deep Neural Nets, Deep Pyramids CNN, GRU/LSTM+CNN, Hinton's Capsule Net

Others: Light Gradient Boosting Machine, SVM with Naive Bayes features, Factorized machine, logistic regression.

Neural nets models perform pretty well perhaps because this is an NLP task.

 - Model stacking

OK, apparently this is the most important part. There are so many techniques about stacking. The popular one is to use the out-of-fold predictions, i.e. maintaining the same fold split for different models and use those out-of-fold predictions on the validation data as features to do the second level models. Some people call those out-of-fold predictions meta-features and therefore meta-learning. This provides a way for end-to-end learning even without the need for feature engineering. A good example is the code developed by the #1 kaggler Kaz-nova - StackNet. Certainly he used this strategy angin to achieve top performance. I personally didn't find this two level stacking very useful in this competition. Yes, it's true that with the result from the two-level stacking I can get about top 8% and a Bronze medal with a XGBoost model in the second level. But I find a useful strategy to stack the first level results nicely, that is using Monte Carlo simulation to find weights for different models. Since we basically use all the out-of-fold predictions as training data in this way, we need to be careful not to overfit. In my experience, as long as both the local CV and the public leader board increases together, it should be fine. 

 - After thoughts

 After reading several winning solutions, again I felt that the feature engineering is the key that separates people apart. For example, the [first place solution](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557), apart from the fact that they have 6 GPUs that allowed them to try ideas really fast, they also have many interesting ideas. Since the comments are not in English only, they use machine translation to translate the comments first to a middle language, such as German and Spanish, then translate them back to English. This is like a data augmentation. They also use word embeddings from other languages to train with the translated middle language. Note that they do this on both training data and test data. They train their mdoels on those augmented data separately and simply average the results. This technique alone boosts their performance a lot. In addition, they use something called pseudo-labeling. Essentially they pick a good model and use some of the highly probable predicted test data as training data. Some people argues the reason that it works so well in this particular competition is because our AUC is quite high, ~0.987. I can see this reasoning makes sense. Also, they use many word embeddings, basically as many as they could find online. They didn't use second level ensembling like the #1 kaggler Kaz-nova usually dose. They also only used neural nets models, RNNs, CNNs and their combinations. 

One more thing that I find interesting is how people concatenate differenct word embeddings. I thought about this but didn't have time to try out this idea.



