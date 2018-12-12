---
layout: default
---

# Summary

Content recommendation is one of the most active areas of contemporary data science research. Techniques broadly fall within two categories (with some hybrid approaches that combine the two): collaborative filtering and content-based recommendation. Generally, collaborative filtering utilizes data on existing users’ preferences to identify similar users or content, and then makes recommendations based on the similar content or on the content selected by similar users. Content-based recommendation, on the other hand, relies on the underlying features of the content, taking a hedonic approach to identifying songs or movies, depending on the context, that a given user may enjoy.

Our task at hand was to create a system for recommending songs for Spotify users based on existing playlist data. In order to accomplish this goal, we relied on a dataset of a million playlists, of which we sampled 2,000 given the computational complexity of some of our models. The problem differs from other types of recommendation engines in that we do not have an ordinal ranking of user’s tastes. In contrast, a number of authors have tackled the question of movie recommendations, where they are able to make use of users’ revealed preferences from 1-to-5 rankings. Our data does not have any way of revealing when users dislike a song, since the decision not to play a song out of dislike and the decision not to play it out of ignorance are indistinguishable.

We experimented with both collaborative filtering and content-based techniques, and found generally more robust performance from the collaborative filtering methods. For content-based techniques, we generated data through natural language processing (NLP) based on song lyrics, and accessed audio feature metadata provided through Spotify’s API. For collaborative filtering, the techniques tested included SVD matrix factorization, distance-based techniques (Pearson correlation and cosine-distance kNN), and neural networks. Where applicable, we tested different flavors of these techniques, including song-song distance as well as playlist-playlist distance. For our neural networks, we found greater predictive success by using artist-playlist pairs rather than song-playlist pairs, as this helps mitigates against sparsity in the data, a recurring challenge throughout our analysis.

For validation, we grappled with the normative goals of our analysis, as we want to present users with songs they are likely to enjoy, but not merely the same artists and songs they are already listening to. We settled on genre-based validation, although this technique was not perfect, as Spotify provides genre data at a granular level that sometimes created effective false negatives during validation. Overall, through the more robust models we were able to consistently achieve cross-validation scores in the range of 0.4-0.5 based on the mode of recommended songs’ genre being equal to the mode of existing songs’ genres.

![001](images/001-intro.jpeg) | ![002](images/002-intro.jpeg)
*Spotify currently suggests songs to listeners based on what they already listen to.*

## Literature Review

Before trying various data extraction and modeling techniques, we looked at what existing literature exists on using these techniques for recommendation systems, may they be for music, books, movies, or anything else. We also tried to see if existing literature supports our own findings.

### Collaborative Filtering
- P. Parhi, A. Pal and M. Aggarwal. A survey of methods of collaborative filtering techniques. International Conference on Inventive Systems and Control, 2017.
- X. Su and T.M. Khoshgoftaar. A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009.
- N. Becker. Music recommendations with collaborative filtering and cosine distance. Personal blog, https://beckernick.github.io/music_recommender/
- M.A. Hameed, O.A. Jadaan and S. Ramachandram. Collaborative filtering based recommendation: a survey. International Journal of Computer Science and Engineering, 4(5): 859-876, 2012.

The above papers describe the use of collaborative filtering to develop recommendation systems. Becker describes two types of collaborative filtering: user based and item based, where the former is recommending based on the preferences of other people similar to me and the latter is recommending based on other other people who share the same items as me. For this project, item based collaborative filtering was used as we did not know anything about the individual users.

Su and Khoshgoftaar (2009) identify the following hurdles for collaborative filtering: data sparsity, scalability, synonymy, gray sheep, shilling attacks, and privacy protection. They also weigh the pros and cons of content-based and collaborative filtering, and while it is widely understood that collaborative filtering is a better method, hybrid methods between content-based and collaborative filtering sometimes perform better than purely collaborative filtering techniques. They also go into further detail of the different ways to approach collaborative filtering. Parhi et al. (2017) and Hameed et al. (2012) similarly give overviews of different collaborative filtering techniques and a mathematical justification for them, which is less relevant to us for the purposes for this project.

### Matrix Factorization
- M. Gormley. Matrix factorization and collaborative filtering [Powerpoint slides]. Retrieved from https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture25-mf.pdf
- C. Bacukhage. “k-Means Clustering is Matrix Factorization.” Arxiv, 2015.

The above two works talk about matrix factorization as a type of collaborative filtering. Collaborative filtering takes common interests between two parties A and B, and if they share many commonalities then the assumption is that other items B likes A may also like. This is essentially what matrix factorization does in that it looks at how the matrix of preferences of a person A can be expressed as factors of other preference matrices that may exist among other members of the population. Gormely (2017) lists three types of matrix factorization: Unconstrained, Single Value Decomposition, and Non-Negative matrix factorization. Bacukhage (2015) shows that the constrained matrix factorization problem is really a k-Means clustering problem, and this is a technique we have learned in class.

### kNN
- S. Li. How did we build book recommender systems in an hour part 2 - k nearest neighbors and matrix factorization. Towards Data Science, Medium, 2018, https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-an-hour-part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c

Li (2018) demonstrates how to use k-nearest neighbors (kNN) to produce book recommendations. One important data preprocessing they do is filter the data only to popular books, or those that have been rated many times, in order to gain some sort of statistical significance. This was indeed an issue we faced with the million playlists dataset, in that there were many obscure songs that hardly showed up in other playlists, making collaborative filtering difficult. They use sklearn’s metric attribute and set it to “cosine” as the distance measure based on which the nearest neighbors are determined.

### Neural Networks
- N. Batra. Neural networks for collaborative filtering. Personal blog, https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html
- X. He, L. Liao, H. Zhang, L. Nie, X. Hu and T. Chua. Neural collaborative filtering. International World Wide WEb Conference Committee, 2017.
- K. Hsu, S. Chou, Y. Yang and T. Chi. Neural network based next-song recommendation. Arxiv, 2016.
- S. Huang. Introduction to recommender system part 2: neural network approach. Towards Data Science, Medium, 2018, https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7

The above papers describe the neural network approach to collaborative filtering and its applications to recommender systems. Batra (2017) presents the neural network architecture discussed in He et al. (2017), and also shows the Python implementation of such architectures. Hse et al. (2016) proposes a neural network framework for song recommender systems in particular, and claims to be the first paper to do so. Huang (2018) explores some neural network implementation, but finds that SVD factorization performs quite well.

### Content-based
- L.M. de Campos, J.M. Fernandez-Luna, J.F. Huete and M.A. Rueda-Morales. Combining content-based and collaborative recommendations: a hybrid approach based on Bayesian networks. International Journal of Approximate Reasoning, 51(7): 785-799, 2010.

Campos et al. (2010) presents a Bayesian approach to combining content-based and collaborative features into a recommender system. It uses probabilistic reasoning to compute probability distribution over IMDB movie ratings. Overall, they found that the hybrid approach performed better than either of the two approaches separately.


Introduction | [EDA](./eda.html) | [Models](./models.html) | [Conclusions and References](./conclusions.html)
