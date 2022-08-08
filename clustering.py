import pandas as pd
import nltk
import logging
logging.basicConfig(filename="clustering.log", level=logging.INFO, format='%(message)s')
nltk.download('wordnet')
nltk.download('omw-1.4')

class Kmeans:

    def __init__(self,input_data) :
        self.input_data = input_data

    def sum_of_squared_error(self,k_clusters, centroids):
        sse = 0
        for i in centroids.keys():
            for j in list(k_clusters[i]):
                sse += self.jaccard_dissimilarity(centroids[i],j)**2
        return sse

    def text_lemmatization(self,text):
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        word_list = []
        for word in w_tokenizer.tokenize(text):
            lemmatizer.lemmatize(word)
            word_list.append(lemmatizer.lemmatize(word))
        return word_list

    def jaccard_dissimilarity(self,word_1,word_2):
        word_1 = set(word_1)
        word_2 = set(word_2)
        union = list((word_1 | word_2))
        union_length = len(union)
        intersection = list((word_1 & word_2))
        intersection_length = len(intersection)
        return 1-(intersection_length/union_length)

    def preprocess(self):
        self.input_data = self.input_data.drop(self.input_data.columns[[0, 1]], axis=1)
        self.input_data['original'] = self.input_data['tweet'].copy()
        self.input_data['tweet'] = self.input_data['tweet'].str.replace('(@\w+.*?)', "")
        self.input_data['tweet'] = self.input_data['tweet'].replace({'#': ''}, regex=True)
        self.input_data['tweet'] = self.input_data['tweet'].str.rsplit('http').str[0]
        self.input_data['tweet'] = self.input_data['tweet'].str.lower()
        self.input_data['tweet'] = self.input_data.tweet.apply(self.text_lemmatization)
        #return  self.input_data

    def check_if_centroids_updated(self,number_of_clusters,centroids,recomputed_centroids):
        i = 0
        while i < number_of_clusters:
            if(list(centroids.values())[i] != list(recomputed_centroids.values())[i]):
                return True
                break
            i += 1
        return False

    def update_centroids_of_cluster(self,k_clusters,recomputed_centroids):
        #print('k_clusters --->> ',k_clusters)
        #print('recomputed_centroids --->>',recomputed_centroids)
        for i in k_clusters:
            cluster = k_clusters[i]
            dists_in_cluster = []
            for element_1 in cluster:
                #print('element_1 -->> ', element_1)
                if len(element_1) != 0:
                    dist = []
                    for element_2 in cluster:
                        distance = self.jaccard_dissimilarity(element_1, element_2)
                        #print('jaccard_dissimilarity --->>',distance)
                        dist.append(distance)
                    tota_jaccard_distance = sum(dist)
                    #print('tota_jaccard_distance --->>',tota_jaccard_distance)
                    dists_in_cluster.append(tota_jaccard_distance)
                    #print('dists_in_cluster --->>',dists_in_cluster)
            minimum_value = min(dists_in_cluster)
            #print('minimum_value --->>',minimum_value)
            index = dists_in_cluster.index(minimum_value)
            #print('index --->>',index)
            recomputed_centroids[i] = cluster[index]
            #print('recomputed_centroids --->>',recomputed_centroids)


    def kmeans_clustering_jaccard_distance(self,number_of_clusters, centroids=None):
        '''
        Initial We choose cluster centroids by random sampling. Then we moved all points into nearest centroid cluster.
        step-2 : Updating centroid - finding the record in cluster which have the minimum distance with all the cluster member and make it as new centoid as cluster
        step-3 : If cluster is updated then repeat the step-2 gain till centroid doesn't change

        Input : K - number of clusters
        Centroid : None/ dict[k]:tweet

        output : None 
        '''
        self.input_data = self.input_data.sample(frac=1).reset_index(drop=True)
        #print('centroids -->',centroids)
        if centroids == None:
            centroids = {}
            i = 0
            while len(centroids) != number_of_clusters:
                if self.input_data['tweet'][i] not in list(centroids.values()):
                    centroids[i] = self.input_data['tweet'][i]
                    i +=1
                    #print(len(centroids))
        k_clusters = {}
        recomputed_centroids = {}
        for key in range(number_of_clusters):
            k_clusters[key] = []
            recomputed_centroids[key] = []
        #print("centroids ",centroids)
        for word in self.input_data['tweet']:
            dist = []
            for key in centroids:
                dissimilarity = self.jaccard_dissimilarity(word, centroids[key])
                dist.append(dissimilarity)
            #print("dist ",dist)
            min_distance = min(dist)
            min_dist_index = dist.index(min_distance)
            #print('min_dist_index -->',min_dist_index)
            k_clusters[min_dist_index].append(word)
            #print('k_clusters -->',k_clusters)
            
        self.update_centroids_of_cluster(k_clusters,recomputed_centroids)
        clusters_updated = self.check_if_centroids_updated(number_of_clusters,centroids,recomputed_centroids)
        #print('clusters_updated -->',clusters_updated)

        if clusters_updated == True:
            print("clusters_updated, iterating to update again.")
            logging.info("clusters_updated, iterating to update again.")
            centroids = recomputed_centroids.copy()
            self.kmeans_clustering_jaccard_distance(number_of_clusters, centroids)
        else:
            print("Centroid didn't change, sum_of_squared_error = ", self.sum_of_squared_error(k_clusters, centroids))
            print(f"Total {number_of_clusters} Clusters")
            print(" cluster number : # of tweets")
            
            sse = self.sum_of_squared_error(k_clusters, centroids)
            logging.info(f"Centroid didn't change, sum_of_squared_error = {sse}")
            logging.info(f"Total {number_of_clusters} Clusters")
            logging.info("cluster number : # of tweets")
            
            for i in range(number_of_clusters):
                print(i+1, ":", len(k_clusters[i]),"tweets")
                logging.info(f"{i+1}, : , {len(k_clusters[i])}, tweets")

        return None

    def fit(self):
        self.preprocess()
        k_values = [3,5,7,11,13,17,19]
        #k_values = [5]
        for number_of_cluster in k_values:
            self.kmeans_clustering_jaccard_distance(number_of_cluster, centroids=None)

def get_data(URL):
    download_data = pd.read_csv(URL,names=['tweet_id', 'data_time', 'tweet'], sep='|')
    return download_data

def main():
    #download_data = get_data('./usnewshealth.txt')
    download_data = get_data('https://drive.google.com/uc?id=10LtxCuotJ44_xBJDtpV0v7CBhLwzVNkb')
    
    print('sample 5 record of data --->>>>>',download_data.head(5))
    logging.info(download_data.head(5).to_string())
    
    kmeans = Kmeans(download_data)
    kmeans.fit()
    
if __name__ == "__main__":
    main()