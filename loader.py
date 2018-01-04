from sklearn.externals import joblib
from keras.models import load_model
from keras.models import model_from_json

# Load transformers and classifiers
print("Loading - Links CV ..")
cv_links = joblib.load('links/cv_links.joblib.pkl')

print("Loading - Hashtag CV ..")
cv_hashtags = joblib.load('hashtag/cv_hashtags.joblib.pkl')

print("Loading - Mentions CV ..")
cv_mentions = joblib.load('mention/cv_mentions.joblib.pkl')

print("Loading - Friends CV ..")
cv_friends = joblib.load('ids/cv_ids.joblib.pkl')

print("Loading - Tweet CV ..")
cv_tweets = joblib.load('tweets/cv_tweets.joblib.pkl')

print("Loading - Bio CV ..")
cv_bios = joblib.load('bio/cv_bio.joblib.pkl')

print("Loading - Links LB ..")
lb_links = joblib.load('links/lb_links.joblib.pkl')

print("Loading - Hashtag LB ..")
lb_hashtags = joblib.load('hashtag/lb_hashtag.joblib.pkl')

print("Loading - Mentions LB ..")
lb_mentions = joblib.load('mention/lb_mentions.joblib.pkl')

print("Loading - Friends LB ..")
lb_friends = joblib.load('ids/lb_ids.joblib.pkl')

print("Loading - Tweet LB ..")
lb_tweets = joblib.load('tweets/lb_tweets.joblib.pkl')

print("Loading - Bio LB ..")
lb_bios = joblib.load('bio/lb_bio.joblib.pkl')

print("Loading - Links classifier ..")
json_data = open('links/Linksmodel.txt').read()
linksclf = model_from_json(json_data)
linksclf.load_weights('links/Links3layer256.h5')

print("Loading - Hashtag classifier ..")
json_data = open('hashtag/Hashtagsmodel.txt').read()
hashtagclf = model_from_json(json_data)
hashtagclf.load_weights('hashtag/Hashtags_weights.h5')

print("Loading - Mentions classifier ..")
json_data = open('mention/Mentionsmodel.txt').read()
mentionclf = model_from_json(json_data)
mentionclf.load_weights('mention/Mentions_weights.h5')

print("Loading - Friends classifier ..")
json_data = open('ids/Idsmodel.txt').read()
friendsclf = model_from_json(json_data)
friendsclf.load_weights('ids/Ids_weights.h5')

print("Loading - Tweet classifier ..")
json_data = open('tweets/Tweetsmodel.txt').read()
tweetclf = model_from_json(json_data)
tweetclf.load_weights('tweets/Tweets_weights.h5')

print("Loading - Bio classifier ..")
json_data = open('bio/Biomodel.txt').read()
bioclf = model_from_json(json_data)
bioclf.load_weights('bio/bio3layer256.h5')
