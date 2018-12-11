---
title: Spotify Project Homepage
notebook: index.md
nav_include: 0
---

Team: Apaar Bansal, Jacob Kohn, Alan Sage (Group No. 58)
TF: Yujiao Chen

#Introduction

**Summary**
Content recommendation is one of the most active areas of contemporary data science research. Techniques broadly fall within two categories (with some hybrid approaches that combine the two): collaborative filtering and content-based recommendation. Generally, collaborative filtering utilizes data on existing users’ preferences to identify similar users or content, and then makes recommendations based on the similar content or on the content selected by similar users. Content-based recommendation, on the other hand, relies on the underlying features of the content, taking a hedonic approach to identifying content that a given user may enjoy.

Our task at hand was to create a system for recommending songs for Spotify users based on existing playlist data. In order to accomplish this goal, we relied on a dataset of a million playlists, of which we sampled 2,000 given the computational complexity of some of our models. The problem differs from other types of recommendation engines in that we do not have an ordinal ranking of user’s tastes. In contrast, a number of authors have tackled the question of movie recommendations, where they are able to make use of users’ revealed preferences from 1-to-5 rankings. Our data does not have any way of revealing when users dislike a song, since the decision not to be play a song out of dislike and the decision not to play it out of ignorance are indistinguishable.

We experimented with both collaborative filtering and content-based techniques, and found generally more robust performance from the collaborative filtering methods. For content-based techniques, we generated data through natural language processing (NLP) based on song lyrics, and accessed audio feature metadata provided through Spotify’s API. For collaborative filtering, the techniques tested included SVD matrix factorization, distance-based techniques (Pearson correlation and cosine-distance kNN), and neural networks. Where applicable, we tested different flavors of these techniques, including song-song distance as well as playlist-playlist distance. For our neural networks, we found greater predictive success by using artist-playlist pairs rather than song-playlist pairs, as this helps mitigates against sparsity in the data, a recurring challenge throughout our analysis.

For validation, we grappled with the normative goals of our analysis, as we want to present users with songs they are likely to enjoy, but not merely the same artists and songs they are already listening to. We settled on genre-based validation, although this technique was not perfect, as Spotify provides genre data at a granular level that sometimes creates effective false negatives. Overall, through the more robust models we were able to consistently achieve cross-validation scores in the range of 0.4-0.5 based on the mode of recommended songs’ genre being equal to the mode of existing songs’ genres.

**Literature Review**
Insert Literature Review here.


>here is a quote

Here is *emph* and **bold**.

Here is some inline math $\alpha = \frac{\beta}{\gamma}$ and, of-course, E rules:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu}  = 8 \pi T_{\mu\nu} . $$
