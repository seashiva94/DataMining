import sys
import numpy as np
import pandas as pd


def euc(X,y):
    """
    return eiclidean distance of all points in x from point y
    """
    X = np.array(X)
    y = np.array(y)
    return np.sqrt(np.sum( (X-y)**2 , axis=1))


def cosine(X,y):
    """
    return cosine distance of all points in x from point y
    """
    # numpy on hulk doesnt support axis in norm
    #print y
    X = np.array(X)
    y = np.array(y)
    normx = np.array([np.linalg.norm(x) for x in X])
    normy = np.linalg.norm(y)
    z = normx*normy
    cos = np.dot(X,y)
    return 1.0 - cos/z


def manhattan(X,y):
    """
    return manhattan distance of all points in x from point y
    """
    X = np.array(X)
    y = np.array(y)
    diff = X-y
    return np.sum(np.abs(diff), axis = 1)


def hamming(X,y):
    """
    return Hamming disance of all points in x to point y
    """
    print "called"
    X = np.array(X)
    y = np.array(y)
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    diff = X-y
    return np.sum(diff, axis = 1)


class recommender:
    def __init__(self, distance=cosine, k=5):
        self.distance = distance
        self.k = k
    
    def predict_rating(self, user, movie):
        """
        predict movies rating  by  user
        in test data
        user-id: row index in matrix
        movie-id: col index in matrix
        """
        
        ids = np.where(self.matrix[:, movie] > 0)[0]
        if len(ids) == 0:
            return 0
        reviewers = self.matrix[ids]
        curr_user = self.matrix[user]
        dists = self.distance(reviewers, curr_user)
        
        k = np.min([self.k, ids.shape[0]])
        sorted_k_user_ids = np.argsort(dists)[:k]
        ratings = reviewers[sorted_k_user_ids][:,movie]
        return np.round(np.mean(ratings))

    def evaluate_model(self, training_data, test_data, read=True, filename = "matrix.txt"):
        """
        evaluate the model based on the given
        error function
        """
        self.matrix = training_data.values
        MAD = 0.0
    
        for i in range(test_data.shape[0]):
            user = test_data.iloc[i,0] -1
            movie = test_data.iloc[i,1] -1
            target_rating = test_data.iloc[i,2]

            predicted_rating = self.predict_rating(user, movie)
            diff = np.abs(predicted_rating - target_rating)
            MAD += diff
        
        return MAD/test_data.shape[0]


    def get_naive_pred(self, user, movie ):
            idx = np.where(self.matrix[:,movie] > 0)
            predicted= np.mean(self.matrix[idx])
            return predicted

    def get_naive_rating(self, test_data):
        MAD = 0.0
        for i in range(test_data.shape[0]):
            user = test_data.iloc[i,0] -1
            movie = test_data.iloc[i,1] -1
            target_rating = test_data.iloc[i,2]
            
            idx = np.where(self.matrix[:,movie] > 0)[0]
            if len(idx) == 0:
                continue 
            predicted_rating = np.mean(self.matrix[idx])
            diff = np.abs(predicted_rating - target_rating)
            MAD += diff
            print "current _MAD = ", MAD/i
        return MAD/test_data.shape[0]



class recommender_v2(recommender):
    def __init__(self, k=5, distance = cosine):
        self.k = k
        self.distance = distance
        movie_file = "/l/b565/ml-100k/u.item"
        self.genres = pd.read_csv(movie_file, header= None, sep = "|")
        self.all_movie_ids = self.genres[0].values
        self.genres = self.genres.iloc[:,5:].values
        user_file = "/l/b565/ml-100k/u.user"
        self.users = pd.read_csv(user_file, header= None, sep = "|").drop([0,3,4], axis = 1)
        self.users = pd.get_dummies(self.users).values
        
    def predict_rating(self, user, movie):
        movie_genres = self.genres[movie]
        dists = self.distance(self.genres, movie_genres)
        closest_movie_idxs = np.argsort(dists)[:self.k]        
        movie_ids = self.all_movie_ids[closest_movie_idxs] 
        cols = np.where(np.in1d(self.training_movie_ids, movie_ids))[0]
        subset = self.matrix[:,cols]
        
        user_dists = self.distance(subset, subset[user])

        closest_user_idxs = np.argsort(user_dists)[:self.k]
        subset = subset[closest_user_idxs,:]
        user_ratings = subset[np.where(subset > 0)]
        return np.round(np.mean(user_ratings)*2)/2
    
    def evaluate_model(self, training_data, test_data):
        self.matrix = training_data.values
        print self.matrix.shape, self.users.shape
        self.matrix = np.append(self.matrix, self.users, axis=1)
        self.training_movie_ids = training_data.columns.values
        MAD = 0.0
        for i in range(test_data.shape[0]):
            user = test_data.iloc[i,0] -1
            movie = test_data.iloc[i,1] -1
            target = test_data.iloc[i,2] -1
            predicted = self.predict_rating(user, movie)
            if np.isnan(predicted):
                continue
            MAD += np.abs(predicted - target)
            print i, MAD/i
        return MAD/test_data.shape[0]

class recommender_v3:
    def __init__(self, k = 5, distance  = cosine):
        self.k = k
        self.distance = distance

        movies_file = "/l/b565/ml-10M100K/movies.dat"
        movies = pd.read_csv(movies_file, header =None, sep = "::", engine ="python")
        self.movie_ids = movies[0].values # as movie ids are not continuous
        self.genres = movies[2]
        self.genres = self.genres.apply(lambda x: x.strip().split("|"))
        self.genres = pd.get_dummies(self.genres.apply(pd.Series).stack()).sum(level = 0).values
        print self.genres[1]
        # now we have one hot genres of each movie
        
    def predict_rating(self, user, movie):
        """
        training and test data are data frames
        may need movie_ids to see if present
        """
        rating = 0
        if user > 0 :
            movie_idx = np.where(self.movie_ids == movie)[0][0]
            movie_genre = self.genres[movie_idx]

            distances = self.distance(self.training_movie_genres, movie_genre)
            k = np.min([self.k, distances.shape[0]])

            sorted_movie_idxs = np.argsort(distances)[:k]
            actual_movie_idxs = self.training_movs[sorted_movie_idxs] # basically column numbers
            cols = np.where(np.in1d(self.training_data.columns.values, actual_movie_idxs)) 
            subset_data = self.training_data.values[:, cols]
            user_rating_data = subset_data[np.where(subset_data > 0)]
            rating = np.mean(user_rating_data)
        return np.round(rating*2)/2

    def evaluate_model(self, training_data, test_data):

        self.training_data = training_data
        self.training_movs = np.array(training_data.columns)
        self.training_movie_idxs = np.where(np.in1d(self.movie_ids, self.training_movs))[0]
        self.training_movie_genres = self.genres[self.training_movie_idxs]

        MAD = 0.0
        for i in range(test_data.shape[0]):
            print i
            user = test_data.iloc[i,0]
            movie = test_data.iloc[i,1]
            target = test_data.iloc[i,2]
            predicted = self.predict_rating(user, movie)
            MAD += np.abs(target - predicted)            
            print target, predicted
            print "current_mad = ", MAD/i
        return MAD/test_data.shape[0]


if __name__ == "__main__":
    
    distances = {"euc":euc, "manhattan": manhattan, "cosine":cosine, "hamming":hamming}
    classes = {'1': recommender, '2':recommender_v2, '3':recommender_v3}
    if len(sys.argv) < 6:
        print "USAGE: python recommender.py <training file> <test file> <distance> <k> <part>"
        print "training_file: if part = 1 or part = 2 training file can be ui.base [i = 1,2,3,4]"
        print "\t if part = 3 training file can be ri.train [i = 1,2,3,4]"
        print "test_file: if part = 1 or part = 2 test file can be ui.test [i = 1,2,3,4]"
        print "\t if part = 3 test file can be ri.test [i = 1,2,3,4]"
        print "distance : [euc, manhattan, cosine]"
        print "k: [integer value like 5,10,30,100...]"
        print "part: [1,2,3] what part of the question"
        sys.exit()



    train_file = sys.argv[1]
    test_file = sys.argv[2]
    distance = distances[sys.argv[3]]
    k = int(sys.argv[4])
    part = sys.argv[5]

    if part == '3':
        prefix = "/l/b565/ml-10M100K/"
        sep = "::"
    else:
        prefix = "/l/b565/ml-100k/"
        sep = "\t"


    train_file = prefix + train_file
    test_file = prefix + test_file
    training_data = pd.read_csv(train_file, header = None, sep = sep, engine = "python")
    test_data = pd.read_csv(test_file, header = None, sep = sep, engine="python")    
    
    recommender = classes[part](k = k, distance = distance)
    
    training_data = training_data.drop(3, axis = 1)
    training_data = training_data.drop_duplicates()
    training_data.columns = ["user_id", "movie_id", "rating"]
    training_data = training_data.pivot(index = "user_id", columns = "movie_id", values = "rating")
    
    mad = recommender.evaluate_model(training_data, test_data)
    print "mad = ", mad
    
