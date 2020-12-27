from bot import telegram_bot
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
mybot = telegram_bot("config.txt")
# def get_title_from_index(index):
# 	return df[df.index == index]["title"].values[0]

# def get_index_from_title(title):
# 	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File

update_id = None

def recom():
	def get_title_from_index(index):
		return df[df.index == index]["title"].values[0]

	def get_index_from_title(title):
		return df[df.title == title]["index"].values[0]
	df = pd.read_csv("movie_dataset.csv")
	#print df.columns
	##Step 2: Select Features

	features = ['keywords','cast','genres','director']
	##Step 3: Create a column in DF which combines all selected features
	for feature in features:
		df[feature] = df[feature].fillna('')

	def combine_features(row):
		try:
			return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
		except:
			print ("Error:", row)

	df["combined_features"] = df.apply(combine_features,axis=1)

	print ("Combined Features:", df["combined_features"].head())
	print("here")
	##Step 4: Create count matrix from this new combined column
	cv = CountVectorizer()
	print("here")

	count_matrix = cv.fit_transform(df["combined_features"])
	print("here")
	##Step 5: Compute the Cosine Similarity based on the count_matrix
	cosine_sim = cosine_similarity(count_matrix) 
	movie_user_likes = item["message"]["text"]
	# movie_user_likes = "house party 2"
	movie_user_likes_up = movie_user_likes.title()
	print("there")
	## Step 6: Get index of this movie from its title
	try:
		movie_index = get_index_from_title(movie_user_likes_up)
		print("there")
		print('index')
		print(movie_index)
		similar_movies =  list(enumerate(cosine_sim[movie_index]))

		## Step 7: Get a list of similar movies in descending order of similarity score
		sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

		## Step 8: Print titles of first 50 movies
		i=0
		stringmovies=''
		print("your movie: "+movie_user_likes)
		for element in sorted_similar_movies:
				stringmovies += get_title_from_index(element[0])+'\n'
				print (get_title_from_index(element[0]))
				# print(stringmovies)
				i=i+1
				if i>5:
					break
		return stringmovies
	except:
		ex = 'Sorry this movie is not in our data'
		return ex
	

def make_reply(msg):
	if msg is not None:
		return msg

while True:
	print("...")
	updates = mybot.get_updates(offset=update_id)
	updates = updates["result"]
	if updates:
		for item in updates:
			update_id = item["update_id"]
			try:
				message = item["message"]["text"]
				stringmovi = recom()
				print(stringmovi)
			except:
				message = None
			from_ = item["message"]["from"]["id"]
			reply = make_reply(stringmovi)
			mybot.send_message(reply, from_)