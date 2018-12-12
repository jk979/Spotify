---
layout: default
---

# Models

## Singular Value Decomposition (SVD) Matrix Factorization

We began by constructing a number of difference collaborative filtering models. Specifically, we first utilized singular value decomposition (SVD) matrix factorization, which reduces the dimensionality of our data and creates a first matrix of dimension k x u and a second matrix of dimension i x k, where u is the number of users, or in this case playlists, and i is the number of items, or in this case songs. Based on k as a hyperparameter, the SVD algorithm utilizes gradient descent to find the matrix of the specified dimensions which best approximates the training set. In doing so, it makes approximations that are used to identify latent factors in user’s preferences, which can help suggest new songs that a user might actually like to include in their playlist. Our goal is precisely not to fit the model perfectly, as the “misclassifications” of 0s and 1s actually represent recommendations in this context.

```
def matrix_factor_global(k):
    U, sigma, Vt = svds(tracks_frame, k = 50)
    sigma = np.diag(sigma)
    first_product = np.dot(U,sigma)
    all_predicted = np.dot(first_product,Vt)
    pred_df = pd.DataFrame(all_predicted,columns=tracks_frame.columns)
    return pred_df

```

```
def predict_mf(pid,pred_df,tracks_frame,unique_tracks,num_recommendations=10,show_previous=10,verbose=True):
    sorted_predictions = pd.DataFrame(pred_df.iloc[pid,:].sort_values(ascending=False))
    pred_join = pd.merge(sorted_predictions,unique_tracks, how='left', left_on=sorted_predictions.index, right_on='track_uri')
    existing_songs = pd.DataFrame(tracks_frame.iloc[pid,:][tracks_frame.iloc[pid,:]==1])
    existing_join = pd.merge(existing_songs,unique_tracks,how='left',left_on=existing_songs.index,right_on='track_uri')
    pred_join.drop_duplicates()
    if (verbose == True):
        print("Previous Picks:",existing_join.track_name[0:show_previous])
        print("Recommended Songs:",pred_join.track_name[0:num_recommendations])
    return pred_join[0:num_recommendations]
```

![04](images/04-genre-val-accuracy.png)  
*Figure 1*

Based on genre-based validation using a random subset of songs, the optimal value of k appeared to be 10, achieving a validation accuracy of 0.41 *(Figure 1)*. Note that given computational complexity, we chose relatively small validation sets, in this case a random sample equal to 5% of the total playlists. In addition, we also performed qualitative validation by looking at particularly distinctive playlists to best measure if the tool was successfully identifying songs based on intuition. For example, we found the following robust results based on a playlist of Spanish-language songs and a playlist of Christmas songs:

**Christmas Songs**

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| All I Want for Christmas Is You, Santa Baby, Have Yourself A Merry Little Christmas - 1999, Last Christmas, I've Got My Love To Keep Me Warm, White Christmas, It's Beginning to Look a Lot Like Christmas, Winter Wonderland, The Christmas Song (Merry Christmas To You), Little Saint Nick | All I Want for Christmas Is You, It's Beginning To Look A Lot Like Christmas, Holly Jolly Christmas, Have Yourself A Merry Little Christmas, Jingle Bells (feat. The Puppini Sisters), Christmas (Baby Please Come Home), Santa Baby, White Christmas (Duet With Shania Twain), I'll Be Home For Christmas, All I Want For Christmas Is You |
| ![05](images/05-all-i-want.png) | ![06](images/06-buble.png) |

**Spanish-Language Songs**

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| Bella y Sensual, Cuatro Babys, Quiero Repetir, Mera Bebe, En La Intimidad, Vuelva, Snapchat, Felices los 4, Corazon de Seda (feat. Ozuna), Recuerdos | Fanática Sensual, El Perdón, Ginza, 6 AM, Borro Cassette, Dile Que Tu Me Quieres, Bailando - Spanish Version, Yo Te Lo Dije, La Gozadera |
| ![07](images/07-ozuna.png) | ![08](images/08-fanatica.png) |

## Pearson Correlation (Song-Song and Playlist-Playlist)

In addition to implementing the widely used SVD matrix factorisation technique, we also used a basic Pearson coefficient pairwise correlation approach, which does not attempt to identify any latent factors in the data but simply offers recommendations based on correlation with other songs. Note that we are not referring to the actual content of the songs, as this is a pure collaborative filtering technique. Rather, we are referring to the similarity of other songs based on which playlists they mutually appear in. More broadly in the literature, this type of methodology is sometimes referred to as memory-based item-item collaborative filtering. In terms of prediction, our function takes a random sample of songs from a given playlist, finds similar songs, and then identifies the most similar ones.

```
def predict_mbss(pid,tracks_frame,unique_tracks,num_recommendations=10,show_previous=10,n_sample=3,verbose=True):
    existing_songs = pd.DataFrame(tracks_frame.iloc[pid,:][tracks_frame.iloc[pid,:]==1])
    existing_join = pd.merge(existing_songs,unique_tracks,how='left',left_on=existing_songs.index,right_on='track_uri')
    compiled_results = []
    sample_songs = existing_songs.sample(n=n_sample)
    for es in sample_songs.iterrows():
        selected = tracks_frame[es[0]]
        corr_matrix = tracks_frame.corrwith(selected)
        corr_matrix = pd.DataFrame(corr_matrix,columns=['pearsonR'])
        corr_matrix.dropna(inplace=True)
        corr_matrix = corr_matrix.sort_values('pearsonR',ascending=False)[1:num_recommendations] #exclude existing song
        compiled_results.append(corr_matrix)
    corr_results = pd.concat(compiled_results)
    output_res = pd.DataFrame(corr_results.sort_values('pearsonR',ascending=False)[0:num_recommendations])
    output_res = pd.merge(output_res,unique_tracks, how='left', left_on=output_res.index, right_on='track_uri')
    output_res.drop_duplicates()
    if (verbose == True):
        print("Previous Picks:",existing_join.track_name[0:show_previous])
        print("Recommended Songs:",output_res.track_name)
    return output_res
```

In addition, we also implemented a user-user, or playlist-playlist, based approach. Here, similar playlists are identified based on which songs they mutually included and excluded. Specifically, our prediction function first finds the playlist most similar to the input playlist, and then makes a set of random song recommendations from that playlist.

```
def predict_mbuu(pid,tracks_frame,unique_tracks,num_recommendations=10,show_previous=10,num_users=2,verbose=True):
    existing_songs = pd.DataFrame(tracks_frame.iloc[pid,:][tracks_frame.iloc[pid,:]==1])
    existing_join = pd.merge(existing_songs,unique_tracks,how='left',left_on=existing_songs.index,right_on='track_uri')
    tracks_transpose = tracks_frame.transpose()
    corr_matrix = tracks_transpose.corrwith(tracks_frame.iloc[pid,:])
    corr_matrix = pd.DataFrame(corr_matrix,columns=['pearsonR'])
    corr_matrix.dropna(inplace=True)
    corr_matrix = corr_matrix.sort_values('pearsonR',ascending=False)[1:num_users] #exclude existing user
    vals = tracks_frame.iloc[corr_matrix.index[0],:]
    vals = vals[vals == 1]
    recommended = vals.sample(n=min(num_recommendations,len(vals)))
    recommended_uris = pd.DataFrame(recommended.index)
    rec_join = pd.merge(recommended_uris,unique_tracks,how='left',on='track_uri')
    rec_join.drop_duplicates()
    if (verbose == True):
        print("Previous Picks (sample):",existing_join.track_name[0:show_previous])
        print("Recommended Songs",rec_join.track_name)
    return rec_join
```

The Pearson coefficient methodology was generally effective based on genre validation, achieving a similar level of approximately 0.5 based on our small random validation set when using the song-song function. (Again, given computational complexity, it was infeasible to test on a large set). The playlist-playlist (user-user) function performed slightly lower with a genre validation accuracy of 0.3.

A qualitative example demonstrating the capabilities of each recommender is included below for reference:

**Song-Song**

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| Down, Hard to Handle, Broadway, Two Princes, Hooch, Slide, All Over You, Good, Rooster | Jane Says, Longview, Sugarhigh, December, Broadway, Nice to Know You, Far Behind, Stellar - acoostic, Hooch, My Name Is Mud |
| ![09](images/09-alice.png) | ![10](images/10-demon.png) |

**Playlist-Playlist**

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| The House That Built Me, John Cougar John Deere John 3:16, Only Prettier, Springsteen, White Liar, Staring at the Sun, Fast Cars and Freedom, Sippin’ On Fire, Laughed Until We Cried, Feels Like Today | Dance, Dust, Cowboys and Angels, What We Ain’t Got, Aw Naw, Get Your Shine On, Play it Again, Dirt Road Anthem, Blown Away, Til It’s Gone |
| ![11](images/11-rascal.png) | ![12](images/12-kip.png) |

## K-Nearest Neighbors Cosine Distance (Song-Song and Playlist-Playlist)

Subsequently, we implemented a kNN algorithm as well, a similar type of distance-based algorithm, using cosine distance. We utilized cosine distance between different inputs based on precedent in the literature. Again, an item-item recommendation approach was utilized initially, finding the nearest neighbors to a given song based on how many of the same playlists they both appeared in. Subsequently, we expanded to a playlist-playlist based approach, finding similarities between given playlists based on which songs they both included, and then recommending songs from the similar playlists. The song-song genre validation score was 0.6 and the user-user genre validation score 0.33.

```
def predict_knnss(pid,tracks_frame,unique_tracks,num_recommendations=10,show_previous=10,n_sample=5,per_song=5,verbose=True):
existing_songs = pd.DataFrame(tracks_frame.iloc[pid,:][tracks_frame.iloc[pid,:]==1])
    existing_join = pd.merge(existing_songs,unique_tracks,how='left',left_on=existing_songs.index,right_on='track_uri')
    tracks_transpose = tracks_frame.transpose()
    model_knn = NearestNeighbors(metric = 'cosine',algorithm = 'brute')
    model_knn.fit(tracks_transpose)
    selected_playlist = tracks_transpose.iloc[:,pid]
    selected_playlist = selected_playlist[selected_playlist == 1]
    sample_songs = selected_playlist.sample(n_sample)
    master_distances = []
    master_indices = []
    for song in sample_songs.index:
        distances, indices = model_knn.kneighbors(tracks_transpose.loc[song,:].values.reshape(1,-1),n_neighbors=10)
        master_distances.append(distances[0][1:per_song]) #exclude existing
        master_indices.append(indices[0][1:per_song]) #exclude existing
    master_distances = np.asarray(master_distances).flatten()
    master_indices = np.asarray(master_indices).flatten()
    dist_frame = pd.Series(master_distances,index=master_indices)
    dist_index = dist_frame.sort_values(ascending=True)[0:num_recommendations].index
    song_uris = pd.DataFrame(tracks_transpose.iloc[dist_index,:].index)
    output_res = pd.merge(song_uris,unique_tracks, how='left', on='track_uri')
    output_res.drop_duplicates()
    if (verbose == True):
        print("Previous Picks:",existing_join.track_name[0:show_previous])
        print("Recommended Songs",output_res.track_name)
    return output_res
```

```
def predict_knnuu(pid,tracks_frame,unique_tracks,num_recommendations=10,show_previous=10,num_playlists=3,verbose=True):
    existing_songs = pd.DataFrame(tracks_frame.iloc[pid,:][tracks_frame.iloc[pid,:]==1])
    existing_join = pd.merge(existing_songs,unique_tracks,how='left',left_on=existing_songs.index,right_on='track_uri')
    model_knnuu = NearestNeighbors(metric = 'cosine',algorithm = 'brute')
    model_knnuu.fit(tracks_frame)
    selected_playlist = tracks_frame.iloc[pid,:]
    distances, indices = model_knnuu.kneighbors(selected_playlist.values.reshape(1,-1),n_neighbors=10)
    dist_frame = pd.Series(distances[0][1:num_playlists],index=indices[0][1:num_playlists])
    dist_index = dist_frame.sort_values(ascending=True).index    
    vals = tracks_frame.iloc[dist_index[0],:]
    vals = vals[vals == 1]
    recommended = vals.sample(n=min(num_recommendations,len(vals)))
    recommended_uris = pd.DataFrame(recommended.index)
    rec_join = pd.merge(recommended_uris,unique_tracks,how='left',on='track_uri')
    rec_join.drop_duplicates()
    if (verbose == True):
        print("Previous Picks (sample):",existing_join.track_name[0:show_previous])
        print("Recommended Songs",rec_join.track_name)
    return rec_join
```

*Note:  Our recommender system removes exact duplicates from recommendations, but can recommend multiple songs of the same name if have different URIs (meaning if they appear in different albums or are sung by different artists) For presentation here, we have included only the unique song titles.*

**Song-Song**

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| Jasmine Runs Away, Magic Carpet/Parting The Seas, Who Better Than Me - Broadway Cast Recording, The Gospel Truth II, Beyond My Wildest Dreams - Broadway Cast Recording, The Ends of the Earth, If Only (Quarter) - Broadway Cast Recording, Main Title - The Little Mermaid, Aladdin and Cassim Escape, I Want the Good Times Back | Beauty And The Beast (Reprise), Overture, One Jump Ahead (Reprise)|
| ![13](images/13-aladdin.png) | ![14](images/14-beauty.png) |

**Playlist-Playlist**

| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| Lookin Boy - Main Version, Do Ya Head Like This, Scenario - LP Mix, It’s Goin’ Down (feat. Nitti), Beautiful Girls, Sexy Can I feat. Yung Berg, Look At Me Now, SexyBack, Soul Survivor, Gin and Juice | Bossy - Feat. Too $hort, Say My Name, Magnolia, Beautiful Girls, 7 Days, The Sign, Teach Me How to Dougie, Misery Business, Creep, Ice Ice Baby |
| ![15](images/15-ray.png) | ![16](images/16-nelly.png) |

## Neural Networks (Song-Song and Playlist-Playlist)

![17](images/17-model.png)  
*Figure 2*

```
song_input = Input(shape=[1])
song_embedding = Embedding(n_song + 1, n_latent_factors_song)(song_input)
song_vec = Flatten()(song_embedding)

user_input = Input(shape=[1])
user_embedding = Embedding(n_user + 1, n_latent_factors_user)(user_input)
user_vec = Flatten()(user_embedding)

dot = Dot(axes=1)([song_vec, user_vec])
dense_2 = Dense(100,activation='sigmoid')(dot)
dense_3 = Dense(50,activation='sigmoid')(dense_2)
dense_4 = Dense(25,activation='sigmoid')(dense_3)

result = Dense(1, activation='linear')(dense_4)
adam = optimizers.Adam()
model = Model([song_input,user_input], result)
model.compile(optimizer=adam,loss= 'mean_absolute_error')

model.summary()
history = model.fit([item_id,user_id], y_train, epochs=30, verbose=1)
model.save('model.h5')
```

As a last collaborative filtering model, we attempted to implement a neural- network-based approach. The size of the possible playlist-song combinations made training a neural network on the full dataset computationally infeasible. In addition, when we trained on a relatively large subset of songs, even after experimenting with various network architectures, all of which included an embedding layer to identify latent factors, the model typically converged on a constant prediction of 0, namely always assuming that a user would not listen to a song. This makes sense given the relative sparsity of the matrix, as instances when a user does not listen to a song are much more common than vice versa.

```
yhat = model.predict([item_id,user_id])
yhat = pd.DataFrame(yhat)
yhat.columns = ['Yes']
yhat['item_id'] = item_id
yhat['user_id'] = user_id

y_actuals = pd.DataFrame(y_train)
y_actuals.columns=['Yes']
y_actuals['item_id'] = item_id
y_actuals['user_id'] = user_id

#Neural Net Song-User Pairs
def predict_nnsu(pid,x_train,yhat,y_actuals,unique_tracks,verbose=True):
    predictions = yhat[user_id == pid]
    actuals = y_actuals[user_id == pid]
    delta = predictions['Yes'] - actuals['Yes']
    predictions['delta'] = delta
    predictions = predictions.sort_values(ascending=False,by='delta')
    indices_one = predictions.item_id[predictions.delta >= 0.1].values
    pred_tracks = pd.DataFrame([x_train.track_uri[[index]][0] for index in indices_one],columns=['track_uri'])
    pred_join = pd.merge(pred_tracks,unique_tracks, how='left', on='track_uri')
    pred_join.drop_duplicates()
    actuals_yes = actuals[actuals.Yes == 1]
    indices_two = actuals_yes.item_id.values
    actual_tracks = pd.DataFrame([x_train.track_uri[[index]][0] for index in indices_two],columns=['track_uri'])
    actual_join = pd.merge(actual_tracks,unique_tracks, how='left', on='track_uri')
    if (verbose == True):
        print("Predicted Tracks",pred_join.track_name)
        print("Previous Picks:",actual_join.track_name)
    return pred_join
```

Specifically, the architecture we employed began with an embedding layer (similar to matrix factorization) for both the song inputs and playlist inputs. The dot product of these two inputs is then taken, after which we employ a standard artificial neural net architecture, with three hidden layers (100 nodes, 50 nodes, and then 25 nodes), followed by a final activation layer with a single output. We utilized Adam for optimization, which we found to produce the best results and most quickly converge, which is an important concern given the large size of our dataset, even after sampling for only 2,000 playlists. For embedding, we utilized 15 latent factors for both songs and users.

To combat this issue, we first attempted to train the network on a filtered subset of songs with a positive bias, specifically including only songs that occurred in over 30 playlists. Interestingly, we found the model to produce the best results with a linear activation function (the hidden layers used a sigmoid function), which did not make intuitive sense to us given we considered the task to be a classification problem. Nonetheless, we tested our various neural net implementations with a variety of activation functions, and none performed as well as the linear function. We speculate this may be related to the sparsity of the data, and the net being coerced towards predictions of 0.

However, even with these modifications, the first neural-net only produced a genre validation accuracy of 0.07. In many cases, the neural net would not make any recommendations, as we only suggest a song to the user if the model outputs a “1” when the playlist-song pair previously had exhibited a “0” (i.e. the song was not played).

We proceeded to experiment with an artist-playlist neural net, which as discussed in the EDA significantly mitigates against sparsity. We employed the same structure (shown in *Figure 2*) of embedding layers followed by dot product and then three hidden layers. However, in this case, the inputs which were embedded were the artists and playlists.

The results significantly improved by analyzing artists. In this case our predictor function recommends a series of artists based on which artists are included in a given playlist. Based on the artists’ genre, the validation score achieved was 0.22. Example recommendations are shown below.


**Playlist-Song**

| Previous Songs (sample)        | Recommended Songs |
|:-------------|:------------------|
| Planez, Money Longer, That’s What I Like, Rollin, Location, Wild Thoughts, Magnolia, Drowning (feat. Kodak Black) | Congratulations, XO Tour Llif3 |
| ![18](images/18-magno.png) | ![19](images/19-ispy.png) |

**Playlist-Artist**

| Previous Songs (sample)        | Recommended Songs |
|:-------------|:------------------|
| Jake Owen, Dierks Bentley, Blake Shelton, George Strait, Little Big Town, Chris Young | Dustin Lynch, Cole Swindell, Jon Pardi, Chris Stapleton, Eli Young Band, Craig Morgan, Dan + Shay |
| ![20](images/20-blake.png) | ![21](images/21-cole.png) |

## Content-Based Pearson Correlation

We next moved into constructing a series of models based on audio feature metadata accessed via Spotify’s API. Specifically, we developed a model to make recommendations based on acousticness, danceability, energy, liveliness, loudness, speechiness, tempo and valence. As one might perhaps expect, the recommendations did not fully conform to the genre of the existing songs in a playlist, and so it was difficult to qualitatively validate whether the model was performing as expected. In order to ensure our model was performing as intended, we plotted the kernel densities of the distribution of existing/predicted songs compared to the distribution of all songs in the dataset, showing clearly distinct distributions compared to the overall population of songs *(Figure 3)*. In terms of genre-based validation, the model achieve a score of 0.5.


![22](images/22-graph.png)  
*Figure 3*

```
def predict_cbd(pid,tracks_frame,unique_tracks,features_df,num_recommendations=10,show_previous=10,n_sample=3,verbose=True):
    existing_songs = pd.DataFrame(tracks_frame.iloc[pid,:][tracks_frame.iloc[pid,:]==1])
    existing_join = pd.merge(existing_songs,features_df,how='left',left_on=existing_songs.index,right_on='track_uri')
    name_join = pd.merge(existing_songs,unique_tracks,how='left',left_on=existing_songs.index,right_on='track_uri')
    compiled_results = []
    sample_songs = existing_songs.sample(n=n_sample)
    for es in sample_songs.iterrows():
        selected = features_df.loc[es[0],:]
        corr_matrix = features_df.corrwith(selected,axis=1)
        corr_matrix = pd.DataFrame(corr_matrix,columns=['pearsonR'])
        corr_matrix.dropna(inplace=True)
        corr_matrix = corr_matrix.sort_values('pearsonR',ascending=False)[1:num_recommendations] #exclude existing song
        compiled_results.append(corr_matrix)
    corr_results = pd.concat(compiled_results)
    output_res = pd.DataFrame(corr_results.sort_values('pearsonR',ascending=False)[0:num_recommendations])
    output_res = pd.merge(output_res,unique_tracks, how='left', left_on=output_res.index, right_on='track_uri')
    output_res.drop_duplicates()
    if (verbose == True):
        print("Previous Picks:",name_join.track_name[0:show_previous])
        print("Recommended Songs:",output_res.track_name)
    return output_res
```

| Previous Songs (sample)        | Recommended Songs |
|:-------------|:------------------|
| Juke Box Hero, Holiday/Boulevard of Broken Dreams, Girlfriend (German Version), I Don’t Want to Miss a Thing, How You Remind Me, Bad Day, Slide, Hanging By a Moment, Rape Me, Chasing Cars | Right Here, More Than a Feeling, Not Gonna Kill You, Better Off Now, Monday, Runway Houses City Clouds, Boys On the Radio, Sun Is Shining, Move for Me, Threads |
| ![23](images/23-lifehouse.png) | ![24](images/24-joni.png) |

## Content-Based Classification Tree with Boosting

We next explored a classification tree structure for classifying data based on content. Specifically, for every playlist, we calculated the average value for each metadata feature across the playlist. We then calculated the differences for each song relative to the playlist data. The distribution of playlist averages and song-specific values are shown in *Figure 4*, for the most part indicating the centering effect we would expect for the playlist averages. The theory behind our model was that a song whose features are closer to the playlist averages would be more likely to be included in a given playlist.

![25](images/25_1_boosting.png)  
*Figure 4*

```
def predict_tree(pid,allcombos_framet,features_df,unique_tracks,verbose=True):
    allcombos_framet = allcombos_framet[allcombos_framet.index.get_level_values('pid') == allcombos_framet.index.get_level_values('pid').unique()[pid]]
    x_framet = allcombos_framet.index.to_frame()
    x_framet = pd.merge(x_framet,features_df,how='left',on='track_uri')
    x_framet.index = x_framet.track_uri
    y_framet = np.asarray(allcombos_framet)
    pid_avgs = x_framet[y_framet==1].groupby(['pid']).mean()
    pid_avgs.columns = avg_metrics
    x_joint = pd.merge(x_framet,pid_avgs,how='left',on='pid')
    a = x_joint[metrics]
    b = x_joint[avg_metrics]
    b.columns = metrics
    x_in_diffs = a.sub(b).abs()
    predictions = pd.Series(ada_classifier.predict(x_in_diffs),index=allcombos_framet.index.get_level_values('track_uri'))
    actuals = allcombos_framet
    delta = np.asarray(predictions) - np.asarray(actuals)
    predictions = pd.DataFrame(predictions)
    predictions['delta'] = delta
    predictions = predictions.sort_values(ascending=False,by='delta')
    indices_one = pd.DataFrame(predictions[predictions.delta >= 1].index.values,columns=['track_uri'])
    pred_join = pd.merge(indices_one,unique_tracks, how='left', on='track_uri')
    pred_join.drop_duplicates()
    actuals_yes = actuals[actuals == 1]
    indices_two = pd.DataFrame(actuals_yes.index.get_level_values('track_uri').values,columns=['track_uri'])
    actual_join = pd.merge(indices_two,unique_tracks, how='left', on='track_uri')
    if (verbose == True):
        print("Predicted Tracks",pred_join.track_name)
        print("Previous Picks:",actual_join.track_name)
    return pred_join
```

Based on this data, we trained a classification tree to predict if a song would be included in a given playlist given the feature distance from the average. Any songs which had not previously been included but for which the tree predicted a positive binary value of 1 would then be suggested for inclusion. We utilized a boosting approach implemented through AdaBoost with 250 predictors.

![26](images/26-trainingtest.png)  
*Figure 5*

Overall, this model performed poorly. By plotting a training set and test set performance *(Figure 5)*, we were able to see that until a threshold number of predictors, the model simply classified based on the trivial approach of always assuming a song would not be played. When boosting did finally cause the predictor to begin suggesting songs, it only constituted overfitting, as test set performance began declining immediately. The model did not correctly identify the genre of any of the songs during cross-validation (score of 0). In addition, similarly to how we addressed the underperforming song-playlist neural net, we filtered the data here to only include songs which were played over 30 times, helping to mitigate the sparsity issue.

| Previous Songs (sample)        | Recommended Songs |
|:-------------|:------------------|
| No Hands, That’s What I Like, Mask Off, Rolex, No Problem, 2 Phones | Say It, Trap Queen |
| ![27](images/27-rolex.png) | ![28](images/28-trap.png) |

## Lyrics Text-Based Pearson Correlation
We next expanded upon our analysis by incorporating the results of our NLP data into our existing content-based distance model. We built upon the existing audio metadata and added an additional feature for the Compound metrics, which captures the positivity, negativity, and neutrality of the lyrics of a given song. Of the entire sample, we were able to successfully pull lyrics for approximately 18,000 songs, which defined the subset for this model.

The addition of the compound feature did not appear to improve the results when making recommendations based on the Pearson correlation coefficient. During cross-validation, the model achieved a score of only 0.06. Again, because of computational time for calculating distances, it was necessary to utilize a small sample size for cross-validation. Sample qualitative recommendation results are provided.


| Previous Songs        | Recommended Songs |
|:-------------|:------------------|
| Rather Be (feat. Jess Glynne), Summer, Uptown Funk, All About That Bass, Pompeii, Despacito (feat. Daddy Yankee), Wild Thoughts, Sugar (feat. Francesco Yates), Ain’t Nobody (Loves Me Better), Roses | Silvia, Watermelon Crawl, If You Leave, Cannibal, Dirt Road Anthem (feat. Brantley Gilbert), The Power of Love, Caribbean Queen (No More Love on the Run), We Supposed To Be, Walk On Water, U Da Realest |
| ![30](images/30-mars.png) | ![31](images/31-ocean.png) |

## Model Comparison

Overall, based on qualitative analysis of the playlist and performance using cross-validation, we found that the collaborative filtering techniques tended to outperform the content-based recommenders. In addition, those models based on similarities between users without a traditional classification training set tended to show greater robustness. We would consider the various distance-based approaches (Pearson coefficient correlation) and kNN with cosine distance, as well as SVD-based matrix factorization, to show the greatest potential for further exploration. This conclusion is also supported based on the strong qualitative results produced by these models which we have displayed throughout this section. Results are summarized in *Figure 6*.

![29](images/29-cv.png)  
*Figure 6*

[Introduction](./index.md) | [EDA](./eda.md) | Models | [Conclusions and References](./conclusions.html)
