#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis Tools
# 
# Lots of libraries exist that will do sentiment analysis for you. Imagine that: just taking a sentence, throwing it into a library, and geting back a score! How convenient!
# 
# It also might be **totally irresponsible** unless you know how the sentiment analyzer was built. In this section we're going to see how sentiment analysis is done with a few different packages.

# ## Installation
# 
# Use `pip install` two language processing packages, NLTK and Textblob.

# In[1]:


get_ipython().system('pip install nltk textblob')


# ## Tools
# 
# ### NLTK: Natural Language Tooklit
# 
# [Natural Language Toolkit](https://www.nltk.org/) is the basis for a lot of text analysis done in Python. It's old and terrible and slow, but it's just been used for so long and does so many things that it's generally the default when people get into text analysis. The new kid on the block is [spaCy](https://spacy.io/), but it doesn't do sentiment analysis out of the box so we're leaving it out of this right now.
# 
# When you first run NLTK, you need to download some datasets to make sure it will be able to do everything you want.

# In[2]:


import nltk

nltk.download('vader_lexicon')
nltk.download('movie_reviews')
nltk.download('punkt')


# To do sentiment analysis with NLTK, it only takes a couple lines of code. To determine sentiment, it's using a tool called **VADER**.

# In[3]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
sia.polarity_scores("This restaurant was great, but I'm not sure if I'll go there again.")


# Asking `SentimentIntensityAnalyzer` for the `polarity_score` gave us four values in a dictionary:
# 
# - **negative:** the negative sentiment in a sentence
# - **neutral:** the neutral sentiment in a sentence
# - **positive:** the postivie sentiment in the sentence
# - **compound:** the aggregated sentiment. 
#     
# Seems simple enough!
# 
# ### Use NLTK/VADER to determine the sentiment of the following sentences:
# 
# * I just got a call from my boss - does he realise it's Saturday?
# * I just got a call from my boss - does he realise it's Saturday? :)
# * I just got a call from my boss - does he realise it's Saturday? 😊
# 
# Do the results seem reasonable? What does VADER do with emoji and emoticons?

# In[6]:


sia.polarity_scores("I just got a call from my boss - does he realise it's Saturday?")


# In[7]:


sia.polarity_scores("I just got a call from my boss - does he realise it's Saturday? :)")


# In[8]:


sia.polarity_scores("I just got a call from my boss - does he realise it's Saturday? 😊")


# Why do you think it doesn't understand the emoji the same way it understood the emoticon?

# Maybe because emoticon is an image, not a text... Although it says it should in the latest version: 
# https://github.com/cjhutto/vaderSentiment/issues/63 

# #### When VADER was a baby
# 
# As we talked about in class, knowing the dataset a language model was trained on can be pretty important!
# 
# [Can you uncover how VADER was trained by reading its homepage?](https://github.com/cjhutto/vaderSentiment)

# It says, it is especially attuned to microblog-like contexts, such as twitter. 
# 
# My understanding is that it was trained not only on tweets, but on tweets, too. 
# It was trained on:
# 
# - 4,000 tweets pulled from Twitter’s public timeline, plus 200 completely contrived tweet-like texts
# - 5,190 sentence-level snippets from 500 New York Times opinion news editorials/articles
# - 10,605 sentence-level snippets from rotten.tomatoes.com. The snippets were derived from an original set of 2000 movie reviews (1000 positive and 1000 negative) in Pang & Lee (2004); 
# - 3,708 sentence-level snippets from 309 customer reviews on 5 different product from Amazon

# ### TextBlob
# 
# TextBlob is built on top of NLTK, but is infinitely easier to use. It's still slow, but _it's so so so easy to use_. 
# 
# You can just feed TextBlob your sentence, then ask for a `.sentiment`!

# In[9]:


from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer


# In[10]:


blob = TextBlob("This restaurant was great, but I'm not sure if I'll go there again.")
blob.sentiment


# **How could it possibly be easier than that?!?!?** This time we get a `polarity` and a `subjectivity` instead of all of those different scores, but it's basically the same idea.
# 
# Try the TextBlob sentiment tool with another sentence of your own.

# In[13]:


blob = TextBlob("I am positive about this being a very bad choice")
blob.sentiment


# If you like options: it turns out TextBlob actually has multiple sentiment analysis tools! How fun! We can plug in a different analyzer to get a different result.

# In[12]:


blobber = Blobber(analyzer=NaiveBayesAnalyzer())

blob = blobber("This restaurant was great, but I'm not sure if I'll go there again.")
blob.sentiment


# Wow, that's a **very different result.** To understand why it's so different, we need to talk about where these sentiment numbers come from. You can read about [the library behind TextBlob's opinions about sentiment](https://github.com/clips/pattern/wiki/pattern-en#sentiment) but they don't really go into (easily-accessible) detail about how it happens.
# 
# But first: try it with one of your own sentences!

# In[14]:


blob = blobber("I am positive about this being a very bad choice")
blob.sentiment


# ## How were they made?
# 
# The most important thing to understand is **sentiment is always just an opinion.** In this case it's an opinion, yes, but specifically **the opinion of a machine.**
# 
# ### VADER
# 
# NLTK's Sentiment Intensity Analyzer works is using something called **VADER**, which is a list of words that have a sentiment associated with each of them.
# 
# |Word|Sentiment rating|
# |---|---|
# |tragedy|-3.4|
# |rejoiced|2.0|
# |disaster|-3.1|
# |great|3.1|
# 
# If you have more positives, the sentence is more positive. If you have more negatives, it's more negative. It can also take into account things like capitalization - you can read more about the classifier [here](http://t-redactyl.io/blog/2017/04/using-vader-to-handle-sentiment-analysis-with-social-media-text.html), or the actual paper it came out of [here](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf).
# 
# **How do they know what's positive/negative?** They came up with a very big list of words, then asked people on the internet and paid them one cent for each word they scored.
# 
# ### TextBlob's `.sentiment`
# 
# TextBlob's sentiment analysis is based on a separate library called [pattern](https://www.clips.uantwerpen.be/pattern).
# 
# > The sentiment analysis lexicon bundled in Pattern focuses on adjectives. It contains adjectives that occur frequently in customer reviews, hand-tagged with values for polarity and subjectivity.
# 
# Same kind of thing as NLTK's VADER, but it specifically looks at words from customer reviews.
# 
# **How do they know what's positive/negative?** They look at (mostly) adjectives that occur in customer reviews and hand-tag them.
# 
# ### TextBlob's `.sentiment` + NaiveBayesAnalyzer
# 
# TextBlob's other option uses a `NaiveBayesAnalyzer`, which is a machine learning technique. When you use this option with TextBlob, the sentiment is coming from "an NLTK classifier trained on a movie reviews corpus."
# 
# **How do they know what's positive/negative?** Looked at movie reviews and scores using machine learning, the computer _automatically learned_ what words are associated with a positive or negative rating.
# 
# ## What's this mean for me?
# 
# When you're doing sentiment analysis with tools like this, you should have a few major questions: 
# 
# * Where kind of dataset does the list of known words come from?
# * Do they use all the words, or a selection of the words?
# * Where do the positive/negative scores come from?
# 
# Let's compare the tools we've used so far.
# 
# |technique|word source|word selection|scores|
# |---|---|---|---|
# |NLTK (VADER)|everywhere|hand-picked|internet people, word-by-word|
# |TextBlob|product reviews|hand-picked, mostly adjectives|internet people, word-by-word|
# |TextBlob + NaiveBayesAnalyzer|movie reviews|all words|automatic based on score|
# 
# A major thing that should jump out at you is **how different the sources are.**
# 
# While VADER focuses on content found everywhere, TextBlob's two options are specific to certain domains. The [original paper for VADER](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf) passive-aggressively noted that VADER is effective at general use, but being trained on a specific domain can have benefits: 
# 
# > While some algorithms performed decently on test data from the specific domain for which it was expressly trained, they do not significantly outstrip the simple model we use.
# 
# They're basically saying, "if you train a model on words from a certain field, it will be good at sentiment in that certain field."

# ### Comparison chart
# 
# Because they're build differently, sentiment analysis tools don't always agree. Let's take a set of sentences and compare each analyzer's understanding of them.

# In[15]:


import pandas as pd
pd.set_option("display.max_colwidth", 200)

df = pd.DataFrame({'content': [
    "I love love love love this kitten",
    "I hate hate hate hate this keyboard",
    "I'm not sure how I feel about toast",
    "Did you see the baseball game yesterday?",
    "The package was delivered late and the contents were broken",
    "Trashy television shows are some of my favorites",
    "I'm seeing a Kubrick film tomorrow, I hear not so great things about it.",
    "I find chirping birds irritating, but I know I'm not the only one",
]})
df


# In[16]:


def get_scores(content):
    blob = TextBlob(content)
    nb_blob = blobber(content)
    sia_scores = sia.polarity_scores(content)
    
    return pd.Series({
        'content': content,
        'textblob': blob.sentiment.polarity,
        'textblob_bayes': nb_blob.sentiment.p_pos - nb_blob.sentiment.p_neg,
        'nltk': sia_scores['compound'],
    })

scores = df.content.apply(get_scores)
scores.style.background_gradient(cmap='RdYlGn', axis=None, low=0.4, high=0.4)


# Wow, those really don't agree with one another! Which one do you agree with the most? Did it get everything "right?"
# 
# While it seemed like magic to be able to plug a sentence into a sentiment analyzer and get a result back... maybe things aren't as magical as we thought.
# 
# #### Try ten sentences of your own
# 
# Just curious: can you make sentences that specifically "trick" one sentiment analysis tool or another?

# In[19]:


df = pd.DataFrame({'content': [
    "This thing is said to be great, but it's really not",
    "I don't like this wonderful cake",
    "How come this bad experience feels so good",
    "This cat is good, but today it's really bad",
    "Everybody hates this movie, but I adore it"
]})
df


# In[20]:


scores = df.content.apply(get_scores)
scores.style.background_gradient(cmap='RdYlGn', axis=None, low=-0.4, high=0.4)


# ## Review
# 
# **Sentiment analysis** is judging whether a piece of text has positive or negative emotion. We covered several tools for doing automatic sentiment analysis: **NLTK**, and two techniques inside of **TextBlob**.
# 
# Each tool uses a different data to determine what is positive and negative, and while some use **humans** to flag things as positive or negative, others use a automatic **machine learning**.
# 
# As a result of these differences, each tool can come up with very **different sentiment scores** for the same piece of text.

# ## Discussion topics
# 
# The first questions are about whether an analyzer can be applied in situations other than where it was trained. Among other things, you'll want to think about whether the language it was trained on is similar to the language you're using it on.
# 
# **Is it okay to use a sentiment analyzer built on product reviews to check the sentiment of tweets?** How about to check the sentiment of wine reviews?

# I would say reviews are more comparable between themselves, and tweets should be analyzed based on the relevant LM, especially given the fact this data is easily available and such a model does exist. Therefore, one of the main considerations would be to choose the LM the most appropriate for my text data.

# **Is it okay to use a sentiment analyzer trained on everything to check the sentiment of tweets?** How about to check the sentiment of wine reviews?

# The more specific the LM, the better it fits for the specific data. A sentiment analyzer trained on everything will probably give equally okayish results both for tweets and wine reviews. Maybe the logic is also to try out several analyzers on sample data and see which you tink fits the best?

# **Let's say it's a night of political debates.** If I'm trying to report on whether people generally like or dislike what is happening throughout the debates, could I use these sorts of tools on tweets?
# 

# I would try doing so, yes. I like that the goal is not a precise analysis, but rather getting a general understanding.

# We're using the incredibly vague word "okay" on purpose, as there are varying levels of comfort depending on your sitaution. Are you doing this for preliminary research? Are you publishing the results in a journal, in a newspaper, in a report at work, in a public policy recommendation?
# 
# What if I tell you that the ideal of "I'd only use a sentiment analysis tool trained exactly for my specific domain" is both _rare and impractical?_ How comfortable do you feel with the output of sentiment analysis if that's the case?

# I would agree it depends heavily on the purpose and that the purpose should not be publishing the results straight from the analyzer, without verifying them. Rather, the analyzer could inform directions of research. Definitely, the limitations need to be communicated inside the research/article as well.

# As we saw in the last section, **these tools don't always agree with one another, which might be problematic.**

# * What might make them agree or disagree?
# - The differences in how they were created: the LM they were trained on, word (token) selection and the way it was scored. 
# 
# * Do we think one is the "best?"
# - No, it depends on the data you have for analysis and on our purposes.
# 
# * Can you think of any ways to test which one is the 'best' for our purposes?
# - Anyway, we only need it to inform our further research. So it makes sense to compare how they perform on your dataset, spot the differences, explore why those happen. 
