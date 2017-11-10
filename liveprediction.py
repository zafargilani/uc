import tweepy
import json
import threading
from sklearn.externals import joblib
from keras.models import load_model
from keras.models import model_from_json
import configparser

config = configparser.ConfigParser()

#Loading transformers and classifiers
cv_links=joblib.load('links/cv_links.joblib.pkl')
print ("Links CV loaded")

cv_hashtags=joblib.load('hashtag/cv_hashtags.joblib.pkl')
print ("hashtag CV loaded")

cv_mentions=joblib.load('mention/cv_mentions.joblib.pkl')
print ("Mentions CV loaded")

cv_friends=joblib.load('ids/cv_ids.joblib.pkl')
print ("Friends CV loaded")

cv_tweets=joblib.load('tweets/cv_tweets.joblib.pkl')
print ("Tweet CV loaded")

cv_bios=joblib.load('bio/cv_bio.joblib.pkl')
print ("BIO CV loaded")

lb_links=joblib.load('links/lb_links.joblib.pkl')
print ("Links LB loaded")

lb_hashtags=joblib.load('hashtag/lb_hashtag.joblib.pkl')
print ("hashtag LB loaded")

lb_mentions=joblib.load('mention/lb_mentions.joblib.pkl')
print ("Mentions LB loaded")

lb_friends=joblib.load('ids/lb_ids.joblib.pkl')
print ("Friends LB loaded")

lb_tweets=joblib.load('tweets/lb_tweets.joblib.pkl')
print ("Tweet LB loaded")

lb_bios=joblib.load('bio/lb_bio.joblib.pkl')
print ("BIO LB loaded")

json_data=open('links/Linksmodel.txt').read()
linksclf = model_from_json(json_data)
linksclf.load_weights('links/Links3layer256.h5')
print ("Links Classifier loaded")

json_data=open('hashtag/Hashtagsmodel.txt').read()
hashtagclf = model_from_json(json_data)
hashtagclf.load_weights('hashtag/Hashtags_weights.h5')
print ("hashtag Classifier Loaded")

json_data=open('mention/Mentionsmodel.txt').read()
mentionclf = model_from_json(json_data)
mentionclf.load_weights('mention/Mentions_weights.h5')
print ("Mentions Classifier loaded")

json_data=open('ids/Idsmodel.txt').read()
friendsclf = model_from_json(json_data)
friendsclf.load_weights('ids/Ids_weights.h5')
print ("Friends Classifier loaded")

json_data=open('tweets/Tweetsmodel.txt').read()
tweetclf = model_from_json(json_data)
tweetclf.load_weights('tweets/Tweets_weights.h5')
print ("Tweet Classifier loaded")

json_data=open('bio/Biomodel.txt').read()
bioclf = model_from_json(json_data)
bioclf.load_weights('bio/bio3layer256.h5')
print ("BIO Classifier loaded")
    
mapping={1:'PTI',2:'PML-N',3:'MQM',4:'PPP'}
def is_user_valid():
    #stub
    return
def is_user_public():
    #stub
    return

def pull_following(user,allfriends):
    config.read('config.ini')
    CONSUMER_KEY = config['app']['CONSUMER_KEY']
    CONSUMER_SECRET = config['app']['CONSUMER_SECRET']
    ACCESS_TOKEN = config['app']['ACCESS_TOKEN']
    ACCESS_SECRET = config['app']['ACCESS_SECRET']
    print(CONSUMER_KEY)
    #API authentication
    auth = tweepy.OAuthHandler('wAPYPlEpG8RTe4PpyJ0sGau4B', 'DfLpnQrRUG2TEZmB3ovvipn7LV9mQQfqvFjDQo6H5iyx1sVqkE')
    auth.set_access_token('774246744133603328-SRXBwOfBlEILqw9pU3UDicNJHqI9Agz', 'Sy85WcL3hED8O9ocSKbRsy25HdhQMaBeyEJiD1dRG7zsg')
    #auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
    print("Fetching friends")
    for friend in tweepy.Cursor(api.friends_ids,screen_name=user).items():
        # Process the friend here
        allfriends.append((friend))
        if len(allfriends)>=499:
            break
    print("Number of friends pulled:",len(allfriends))
    return allfriends

def pull_tweets(user,alltweets):
    #API authentication
    auth = tweepy.OAuthHandler('wAPYPlEpG8RTe4PpyJ0sGau4B', 'DfLpnQrRUG2TEZmB3ovvipn7LV9mQQfqvFjDQo6H5iyx1sVqkE')
    auth.set_access_token('774246744133603328-SRXBwOfBlEILqw9pU3UDicNJHqI9Agz', 'Sy85WcL3hED8O9ocSKbRsy25HdhQMaBeyEJiD1dRG7zsg')
    #auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
    print ("Fetching tweets")
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = user,count=200)
    #save most recent tweets
    alltweets.extend(new_tweets)
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print (len(alltweets))
        if len(alltweets)>=499:
            return alltweets
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = user,count=200,max_id=oldest,include_rts=True)
	    #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
    return alltweets

def extract_hashtags(alltweets):
    allhashtags=''
    for tweet in alltweets:
        x=tweet._json
        for hashtag in x['entities']['hashtags']:
            
            allhashtags = allhashtags+','+str(hashtag['text'])
    return allhashtags

def extract_mentions(alltweets):
    allmentions=''
    for tweet in alltweets:
            x=tweet._json
            for user_mentions in x['entities']['user_mentions']:
                allmentions=allmentions+'\n'+user_mentions['id_str']
    return allmentions

def extract_ids(allfriends):
    ids=''
    for i in allfriends:
        ids=ids+'\n'+str(i)
    return ids

def extract_links(alltweets):
    all_links=''
    for tweet in alltweets:
        i=tweet._json
        for links in i['entities']['urls']:
            all_links=all_links+'\n'+links['expanded_url']
    return all_links

def extract_tweet(alltweets):
    tweets=''
    for tweet in alltweets:
        x=tweet._json
        text1 = str(x['text'])
        tweets=tweets+','+text1
    return tweets

def extract_bio(alltweets):
    bio=''
    for tweet in alltweets:
        x=tweet._json
        bio = bio+','+str(x['user']['description'])
    return bio

def classify_hashtags(alltweets,Result,):
    global cv_hashtags
    global hashtagclf
    global lb_hashtags
    X_test=extract_hashtags(alltweets)
    print ("Hashtag extracted")
    X_test = cv_hashtags.transform([X_test])
    print ("hashtag applied transform and now predicting")
    X_test=X_test.toarray()
    result=hashtagclf.predict(X_test)
    result2=lb_hashtags.inverse_transform(result)
    result=list(result[0])
    print ("hashtag classification ")
    Result.append("With "+str(max(result)*100)+"% confidence hashtag classification is "+mapping[int(result2)])
    return

def classify_mentions(alltweets,Result,):
    global cv_mentions
    global mentionclf
    global lb_mentions
    X_test=extract_mentions(alltweets)
    print ("Mentions extracted")
    X_test = cv_mentions.transform([X_test])
    print ("Mentions applied transform and now predicting")
    X_test=X_test.toarray()
    result=mentionclf.predict(X_test)
    result2=lb_mentions.inverse_transform(result)
    result=list(result[0])
    print ("Mentions classification ")
    Result.append("With "+str(max(result)*100)+"% confidence Mentions classification is "+mapping[int(result2)])
    return

def classify_friends(allfriends,Result,):
    global cv_friends
    global lb_friends
    global friendsclf
    
    X_test=extract_ids(allfriends)
    print ("Friends extracted")
    X_test = cv_friends.transform([X_test])
    print ("Friends applied transform and now predicting")
    X_test=X_test.toarray()
    result=friendsclf.predict(X_test)
    result2=lb_friends.inverse_transform(result)
    result=list(result[0])
    print ("Friends classification ")
    Result.append("With "+str(max(result)*100)+"% confidence Friends classification is "+mapping[int(result2)])
    return

def classify_tweet(alltweets,Result,):
    global cv_tweets
    global lb_tweets
    global tweetclf
    X_test=extract_tweet(alltweets)
    print ("Tweet extracted")
    
    X_test = cv_tweets.transform([X_test])
    
    print ("Tweet applied transform and now predicting")
    X_test=X_test.toarray()
    result=tweetclf.predict(X_test)
    result2=lb_tweets.inverse_transform(result)
    result=list(result[0])
    print ("Tweet classification ")
    Result.append("With "+str(max(result)*100)+"% confidence Tweets classification is "+mapping[int(result2)])
    return

def classify_bio(alltweets,Result,):
    global cv_bios
    global lb_bios
    global bioclf
    X_test=extract_bio(alltweets)
    print ("BIO extracted")
    
    X_test = cv_bios.transform([X_test])
    
    print ("BIO applied transform and now predicting")
    X_test=X_test.toarray()
    print(X_test.shape)
    
    #with g5.as_default():
    result=bioclf.predict(X_test)
    result2=lb_bios.inverse_transform(result)
    result=list(result[0])
    print ("Bio classification ")
    #result2=clf.predict_proba(X_test)
    #print result2[0][indices[str(result[0])]]
    Result.append("With "+str(max(result)*100)+"% confidence Bio classification is "+mapping[int(result2)])
    return

def classify_links(alltweets,Result,):
    global cv_links
    global lb_links
    global linksclf
    X_test=extract_links(alltweets)
    print ("Links extracted")
    X_test = cv_links.transform([X_test])
    print ("Links applied transform and now predicting")
    X_test=X_test.toarray()
    print(X_test.shape)
    
    result=linksclf.predict(X_test)
    result2=lb_links.inverse_transform(result)
    result=list(result[0])
    print ("Links classification ")
    Result.append("With "+str(max(result)*100)+"% confidence links classification is "+mapping[int(result2)])
    return

def classify(alltweets,allfriends,Result,):
    hashtags=threading.Thread(target=classify_hashtags, args=(alltweets,Result,))
    hashtags.start()

    mentions=threading.Thread(target=classify_mentions, args=(alltweets,Result,))
    mentions.start()

    friends=threading.Thread(target=classify_friends, args=(allfriends,Result,))
    friends.start()

    tweets=threading.Thread(target=classify_tweet, args=(alltweets,Result,))
    tweets.start()

    bios=threading.Thread(target=classify_bio, args=(alltweets,Result,))
    bios.start()

    links=threading.Thread(target=classify_links, args=(alltweets,Result,))
    links.start()

    print ("Working on it")
    friends.join()
    hashtags.join()
    mentions.join()
    tweets.join()
    links.join()
    
    return

def get_prediction(user):
    print ("Entered username: ",user)
    alltweets=[]
    allfriends = []
    Result=[]
    tweets=threading.Thread(target=pull_tweets, args=(user,alltweets))
    tweets.start()
    friends=threading.Thread(target=pull_following, args=(user,allfriends))
    friends.start()
    tweets.join()
    friends.join()
    classify(alltweets,allfriends,Result,)
    return Result



#while(True):
#    user=input('Enter the name of the user: ')
#    print(get_prediction(user))
#    flag=input('More Preds(y/n): ')
#    if(flag=='n'):
#        break
