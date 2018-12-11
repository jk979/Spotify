---
layout: default
---

#EDA
##Data Processing
We began by loading in the playlist datasets and creating various functions to generate a matrix with a line for each song play including its playlist ID. Specifically, we took a sample of 2,000 playlists, representing over 57,000 unique songs. Based on the qualitative findings which we discuss in the Model section, this sample proved satisfactory for generating predictions using many of our selected methodologies. Note that throughout this website, we use the terms “playlist” and “user” interchangeably at times, as without data on how many playlists were created by the same user, the playlist ID takes the role of the user ID in other collaborative filtering contexts.

We next generated a matrix of p playlists by n songs, with a 1 indicating whether that song is present in a playlist. This format was useful for our later collaborative filtering recommendation system model building. The dataset does not have information on which playlists were created by the same users, so we have treated each playlist as equivalent to a user in the traditional parlance of collaborative filtering. This data allowed us to make well-performing models without any knowledge of the underlying content of the songs.

To build upon the existing dataset, we also leveraged the “Spotipy” Python library, a thin client library supporting the features of the Spotify API. This library allows us to submit requests to identify relevant information including song and artist features, and number of plays. We used the library to access genre (discussed below under Validation Data), as well as metadata in the form of audio features, which we then went on to use for constructing a content-based recommendation system. Interestingly, the models built as content recommenders tended to underperform in terms of genre validation compared to the models built using strictly collaborative filtering approaches. We discuss these points further under the Model section.

On a technical note, our current dataset allowed for us to search using both song IDs and song titles. Because many songs could conceivably have the same titles, we relied on the unique song IDs for our database queries in the final project, and then matched the results of our predictor functions with the actual song titles for generating a recommendation output.

##Noteworthy Findings on Song-Playlist Data Structure
Our primary takeaway from our exploratory data analysis was the issue of sparsity in the data. When we began building models, most traditional classification approaches were not able to underperform the trivial strategy of always guessing a song will not be included in a given playlist. When the chances of a song being included in a given playlist are well below 1%, that is not a surprising result.



[back to Homepage](./)
