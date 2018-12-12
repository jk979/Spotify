---
layout: default
---

# Summary

Content recommendation is one of the most active areas of contemporary data science research. Techniques broadly fall within two categories (with some hybrid approaches that combine the two): collaborative filtering and content-based recommendation. Generally, collaborative filtering utilizes data on existing users’ preferences to identify similar users or content, and then makes recommendations based on the similar content or on the content selected by similar users. Content-based recommendation, on the other hand, relies on the underlying features of the content, taking a hedonic approach to identifying songs or movies, depending on the context, that a given user may enjoy.

Our task at hand was to create a system for recommending songs for Spotify users based on existing playlist data. In order to accomplish this goal, we relied on a dataset of a million playlists, of which we sampled 2,000 given the computational complexity of some of our models. The problem differs from other types of recommendation engines in that we do not have an ordinal ranking of user’s tastes. In contrast, a number of authors have tackled the question of movie recommendations, where they are able to make use of users’ revealed preferences from 1-to-5 rankings. Our data does not have any way of revealing when users dislike a song, since the decision not to play a song out of dislike and the decision not to play it out of ignorance are indistinguishable.

We experimented with both collaborative filtering and content-based techniques, and found generally more robust performance from the collaborative filtering methods. For content-based techniques, we generated data through natural language processing (NLP) based on song lyrics, and accessed audio feature metadata provided through Spotify’s API. For collaborative filtering, the techniques tested included SVD matrix factorization, distance-based techniques (Pearson correlation and cosine-distance kNN), and neural networks. Where applicable, we tested different flavors of these techniques, including song-song distance as well as playlist-playlist distance. For our neural networks, we found greater predictive success by using artist-playlist pairs rather than song-playlist pairs, as this helps mitigates against sparsity in the data, a recurring challenge throughout our analysis.

For validation, we grappled with the normative goals of our analysis, as we want to present users with songs they are likely to enjoy, but not merely the same artists and songs they are already listening to. We settled on genre-based validation, although this technique was not perfect, as Spotify provides genre data at a granular level that sometimes created effective false negatives during validation. Overall, through the more robust models we were able to consistently achieve cross-validation scores in the range of 0.4-0.5 based on the mode of recommended songs’ genre being equal to the mode of existing songs’ genres.
[Insert some relevant Spotify image or graphic]

## Literature Review

Before trying various data extraction and modeling techniques, we looked at what existing literature exists on using these techniques for recommendation systems, may they be for music, books, movies, or anything else. We also tried to see if existing literature supports our own findings.

### SVD Matrix Factorization
C. Bacukhage. “k-Means Clustering is Matrix Factorization.” Arxiv, 2015.
N. Becker. Matrix factorization for movie recommendations in Python. Personal blog, https://beckernick.github.io/music_recommender/
M. Gormley. Matrix factorization and collaborative filtering [Powerpoint slides]. Retrieved from https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture25-mf.pdf

### Collaborative Filtering
P. Parhi, A. Pal and M. Aggarwal. A survey of methods of collaborative filtering techniques. International Conference on Inventive Systems and Control, 2017.
X. Su and T.M. Khoshgoftaar. A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009.
N. Becker. Music recommendations with collaborative filtering and cosine distance. Personal blog, https://beckernick.github.io/music_recommender/
M.A. Hameed, O.A. Jadaan and S. Ramachandram. Collaborative filtering based recommendation: a survey. International Journal of Computer Science and Engineering, 4(5): 859-876, 2012.

### kNN
S. Li. How did we build book recommender systems in an hour part 2 - k nearest neighbors and matrix factorization. Towards Data Science, Medium, 2018, https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-an-hour-part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c


### Neural Networks
N. Batra. Neural networks for collaborative filtering. Personal blog, https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html
X. He, L. Liao, H. Zhang, L. Nie, X. Hu and T. Chua. Neural collaborative filtering. International World Wide WEb Conference Committee, 2017.
K. Hsu, S. Chou, Y. Yang and T. Chi. Neural network based next-song recommendation. Arxiv, 2016.
S. Huang. Introduction to recommender system part 2: neural network approach. Towards Data Science, Medium, 2018, https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7

### Content-based
L.M. de Campos, J.M. Fernandez-Luna, J.F. Huete and M.A. Rueda-Morales. Combining content-based and collaborative recommendations: a hybrid approach based on Bayesian networks. International Journal of Approximate Reasoning, 51(7): 785-799, 2010.

### Lyrics-based


[EDA](./eda.html).
[Models](./models.html).
[Conclusions and References](./conclusions.html).
