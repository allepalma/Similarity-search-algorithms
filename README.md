# Similarity-search-algorithms

The present repository contains the implementation of fast algorithms for the detection of pairs of similar objects in massive datasets. With the growing size of
modern data concerning all kinds of digital objects, deploying quick procedures for finding candidate pairs of similar data points is paramount in the development of fast data mining approaches. Through the present work, we aim to illustrate our personal solution to this widespread issue via the implementation of common strategies to approximately single out candidate pairs of associable objects without exhaustively compare all couples. The implemented algorithms have been applied to a rating dataset of Netflix users, nevertheless it can be easily extended to any kind of numerical dataset. The scope of our work is to find similar pairs of users on the base of data displaying how they rated the collected set of movie leveraging three different distance measures: Jaccard similarity, Cosine similarity and Discrete Cosine Similarity.

The constraint of the project implementation are:
- it should provide a solution within 30 minutes each
- pairs are deemed similar if they exceed a Jaccard similarity measure of 0.5 and cosine similarity measures of 0.73

## The dataset
As already mentioned in the introductory section, the dataset we adopted is a downsized version of the one employed originally in the Netflix Challenge. Users were preliminary filtered by removing those who rated less than 300 and more than 3000 movies. The resulting data was collected into a single array object with entries of the form <userID, movieID, rating> containing a total of 65,225,506 records. The IDs are sequences of integers from 1 to the number of users/movies in incremental orders without gaps. In particular, ratings concerning 17,770 movies provided by 103,703 users were present in the dataset. The pool of data we considered was extremely sparse, meaning
that the majority of entries were zero values. Indeed, if we imagine our data represented as a movie x user matrix, 17,770 x 103,703 total entries would be ideally observable. Out of these possible rating slots, only 65,225,506 are occupied, hence approximately the 3.54%. 

The dataset can be downloaded at https://drive.google.com/file/d/1Fqcyu9g6DZyYK_1qmjEgD1LlGD7Wfs5G/view?usp=sharing.

## The creation of a signature matrix
The first step for the approximate isolation of similar pairs is the conversion of the data matrix into a signature matrix. The signature matrix represents the same amount of objects as the original dataset through much shorter vectors. The scope of the signature matrix is to compress the representation of objects in a way that their pairwise similarity is asymptotically preserved, *i.e* similar objects are similar also in the signature representation. The approach for the creation of the signature differs between Jaccard similarity and Cosine similarity. 

 ### Minhashing
There are many ways to address the similarity between couples of users. The first we considered is the Jaccard Similarity. The Jaccard Similarity is an approximate measure of similarity that does not take into account the exact rating that two costumers gave to the items respectively, but the extent to which they rated the same movies. In other words, the Jaccard similarity is larger the more movies in common two user rated.
Mathematically, given two binary sets S1 and S2, where 0 indicates "not rated" and 1 stands for "rated", the
Jaccard similarity is expressed as the size of the intersection of the two sets divided by the size of their union. The value is defined between 0 and 1. 
Thanks to the application of Minhashing, we can compress the highly-dimensional vectors of users to much shorter signature vectors.


Suppose we want signatures of length k and we
are dealing with a movie x user data matrix. Minhashing is realized by first computing k permutations of themovie rows, basically shuffling their indexes. Then, for every permutation i = 1,...,k and for every user j, we fill the entry of the signature matrix (i, j) with the number of the first row that contained a non-empty value for the user j when rows were permuted by i. This leads to a signature matrix that has as many columns as there were users in the original matrix and a total of k rows filled with the number of the first row position that for a user contained a rating in the permutation identified by the row of the signature matrix we consider. Statistically, the probability that two users will have the same value at a certain row is equal to the to their Jaccard similarity. This is valid because, for such a condition to be verified, it means that two users in a permutation have the first non-zero entry at the same row index. The probability for this to happen is equal to the ratio between the number of rows at which both users have non-zero values x and the sum of x with the number of rows in which one of the users reports a rating but the other does not (call this value y). We do not consider entries in which both users have zero values, since they are ignored in the assignment of the signature value. Interestingly, it results that the just described expression x+y equates exactly the formula for the Jaccard similarity.

### Random projections
The cosine similarity measures the similarity between two vectors as the extent to which they point in the same direction. It does this by first quantifying the cosine of the angle between them. Then, when the cosine is calculated, the angle can be computed through the arc-cosine of the cosine measure obtained, which outputs the angle of the two vectors. Mathematically, given two vectors p1 and p2, the cosine similarity can be calculated by first finding the cosine of the two vectors with <img src="https://render.githubusercontent.com/render/math?math=\frac{p_1 \dot p_2}{||p_1|| \dot ||p_2||}">. After computing the cosine, we retrieve the angle through:
<img src="https://render.githubusercontent.com/render/math?math=arccos(\theta)">
and then get the similarity by: <img src="https://render.githubusercontent.com/render/math?math=cosine\_similarity(p1 , p2) = 1 - \frac{\theta}{180}">.


Just like the Jaccard similarity, we implement a hashing algorithm to compress the highly-dimensional vectors of users which is called cosine
sketching. The idea of this algorithm is to iteratively generate k random hyperplanes and create signature vectors of +1 and -1 values depending on whether an object finds itself above or below the random hyperplane selected. To simulate the hyperplane we draw random vectors of uniform values between +1 and -1 and to determine whether a user object lies over or under the hyperplane we check the sign of the dot product between its corresponding vector and the random hyperplane vector itself. 


Two objects agree on a signature row i if they lie on the same side of the ith random vector. By choosing multiple random vectors, and calculating their dot product with user vectors we get resulting vectors filled with +1s and -1s. This is called a sketch. The sketches can be used to estimate the the angle between pairs of rating vectors. This estimate can be quite rough for short signatures, it is therefore important to pick a suffciently large number of random projections. Mathematically, it can be shown that the probability that two users agree on a row of the signature matrix is equal to their cosine similarity. Therefore, the fraction at which two users agree in the signature is a good approximation for their cosine similarity.

## LSH
By developing the Minhashing and sketching processes we are able to represent the user vectors as much shorter iterables without asymptotically compromising the similarity between couples of them. However, to evaluate the Jaccard or cosine similarities of all pairs, we would still need to run an incredibly large number of comparisons,
which would make our algorithm extremely slow. 


In this perspective, we use the Locality Sensitivity Hashing algorithm to focus our attention on candidate pairs of similar users found in an approximate fashion. LSH is based on the so-called banding technique. Once we have created our signature matrix either in terms of sketches
or minhash values, we select two parameters r and b that we use to divide the signature matrix vertically into b bands by r rows each. Consequently, we evaluate for each column and for each band the string obtained by concatenating all the elements of the rows of the band at the level of the considered column. We hash all the
strings for a specific band into buckets by computing a hash function on the relative strings. In this case, the pairs of users that agree on all rows of a band in the signature will end up in the same bucket for that specific band and will be deemed as candidate pairs to examine for the real similarity. For the different bands, we need to construct separate sets of buckets to avoid for users to be sorted to the same bucket if their signatures match at different bands. 


By tuning the number of rows and bands we allow, we will construct a more or less sensitive algorithm to pairs that share a certain level of similarity. Intuitively, more bands with less rows will cause an higher probability of detecting similar pairs since shorter chunks of the signatures have a higher chance to correspond. However, this will also increase the sensitivity to low similarity pairs, which might be deemed candidate and overcrowd the number of comparisons we make. Conversely, longer and less numerous bands will be stricter in the selection of candidate pairs sparing time in terms of amount of conducted comparisons but hindering the sensitivity to lower similarity couples significantly.

