---
layout: default
---

# Models

## Singular Value Decomposition (SVD) Matrix Factorization

We began by constructing a number of difference collaborative filtering models. Specifically, we first utilized singular value decomposition (SVD) matrix factorization, which reduces the dimensionality of our data and creates a first matrix of dimension k x u and a second matrix of dimension i x k, where u is the number of users, or in this case playlists, and i is the number of items, or in this case songs. Based on k as a hyperparameter, the SVD algorithm utilizes gradient descent to find the matrix of the specified dimensions which best approximates the training set. In doing so, it makes approximations that are used to identify latent factors in user’s preferences, which can help suggest new songs that a user might actually like to include in their playlist.

![04](images/04-genre-val-accuracy.png)

Based on genre-based validation using a random subset of songs, the optimal value of k appeared to be 5, achieving a validation accuracy of 0.44. Note that given computational complexity, we chose relatively small validation sets, in this case a random sample equal to 5% of the total playlists. In addition, we also performed qualitative validation by looking at particularly distinctive playlists to best measure if the tool was successfully identifying songs based on intuition. For example, we found the following robust results based on a playlist of Spanish-language songs and a playlist of Christmas songs:

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| All I Want for Christmas Is You, Santa Baby, Have Yourself A Merry Little Christmas - 1999, Last Christmas, I've Got My Love To Keep Me Warm, White Christmas, It's Beginning to Look a Lot Like Christmas, Winter Wonderland, The Christmas Song (Merry Christmas To You), Little Saint Nick | All I Want for Christmas Is You, It's Beginning To Look A Lot Like Christmas, Holly Jolly Christmas, Have Yourself A Merry Little Christmas, Jingle Bells (feat. The Puppini Sisters), Christmas (Baby Please Come Home), Santa Baby, White Christmas (Duet With Shania Twain), I'll Be Home For Christmas, All I Want For Christmas Is You |
| ![05](images/05-all-i-want.png) | ![06](images/06-buble.png) |

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| Bella y Sensual, Cuatro Babys, Quiero Repetir, Mera Bebe, En La Intimidad, Vuelva, Snapchat, Felices los 4, Corazon de Seda (feat. Ozuna), Recuerdos | Fanática Sensual, El Perdón, Ginza, 6 AM, Borro Cassette, Dile Que Tu Me Quieres, Bailando - Spanish Version, Yo Te Lo Dije, La Gozadera |
| ![07](images/07-ozuna.png) | ![08](images/08-fanatica.png) |

![09](images/09-alice.png)
![10](images/10-demon.png)
![11](images/11-rascal.png)
![12](images/12-kip.png)
![13](images/13-aladdin.png)
![14](images/14-beauty.png)
![15](images/15-ray.png)
![16](images/16-nelly.png)
![17](images/17-model.png)
![18](images/18-magno.png)
![19](images/19-ispy.png)
![20](images/20-blake.png)
![21](images/20-cole.png)
![22](images/21-graph.png)
![23](images/23-joni.png)
![24](images/24-graph.png)
![25](images/25-trainingtest.png)
![26](images/26-rolex.png)
![27](images/27-trap.png)
![28](images/28-cv.png)






_yay_

[back](./)
