---
layout: default
---

# Conclusions and preferences

## Future Expansions
While we tested a wide variety of models, we have by no means completed an exhaustive analysis of the various approaches discussed in the literature regarding collaborative-filtering and content recommendation techniques. Many of these techniques are, however, related to the ones we did test out. For example, Bauckhage (2015) showed that k-Means Clustering can be specified as a type of matrix factorization problem as well. There are in addition numerous methodologies for distance-based recommendation, of which we tested only a sample. Experimenting with these similar methodologies could represent a way to improve our results at the margins.

In the future our work could be expanded by a project to analyze stacking techniques and ways to improve the interactions between our various recommenders. In this specific context, this could be a promising avenue, as some of our techniques would be optimized within a specific feature region. For example, we might be able to improve performance of an engine by first using SVD matrix factorization to identify latent factors that approximate genre, and then within those genre categories make specific recommendations based on audio features and mood. These could be more successful than making recommendations based on metadata without any prior knowledge of user’s preferences.

More broadly, we considered validation to be a significant challenge here because our data essentially conflates missing values with negative values. That is, we cannot differentiate between a song which the user actively saw but chose not to play and songs which the user would like to play but was not aware of. This differentiates our specific problem from others discussed in the literature within the context of movie recommendations, where users rank movies on a scale from 1 to 5. In order to achieve a similar approach here, we would need to conduct field testing with our proposed methodologies so that we could collect data on when users are recommended a specific song but choose not to play it, a better proxy metric for dislike.

## References
C. Bacukhage. “k-Means Clustering is Matrix Factorization.” Arxiv, 2015.

N. Batra. Neural networks for collaborative filtering. Personal blog, https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html

N. Becker. Matrix factorization for movie recommendations in Python. Personal blog, https://beckernick.github.io/music_recommender/

N. Becker. Music recommendations with collaborative filtering and cosine distance. Personal blog, https://beckernick.github.io/music_recommender/

L.M. de Campos, J.M. Fernandez-Luna, J.F. Huete and M.A. Rueda-Morales. Combining content-based and collaborative recommendations: a hybrid approach based on Bayesian networks. International Journal of Approximate Reasoning, 51(7): 785-799, 2010.

M. Gormley. Matrix factorization and collaborative filtering [Powerpoint slides]. Retrieved from https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture25-mf.pdf

M.A. Hameed, O.A. Jadaan and S. Ramachandram. Collaborative filtering based recommendation: a survey. International Journal of Computer Science and Engineering, 4(5): 859-876, 2012.

X. He, L. Liao, H. Zhang, L. Nie, X. Hu and T. Chua. Neural collaborative filtering. International World Wide WEb Conference Committee, 2017.

K. Hsu, S. Chou, Y. Yang and T. Chi. Neural network based next-song recommendation. Arxiv, 2016.

S. Huang. Introduction to recommender system part 2: neural network approach. Towards Data Science, Medium, 2018, https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7

S. Li. How did we build book recommender systems in an hour part 2 - k nearest neighbors and matrix factorization. Towards Data Science, Medium, 2018, https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-an-hour-part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c

P. Parhi, A. Pal and M. Aggarwal. A survey of methods of collaborative filtering techniques. International Conference on Inventive Systems and Control, 2017.

X. Su and T.M. Khoshgoftaar. A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009.


_yay_

[back](./)
