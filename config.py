from configparser import ConfigParser
# Initializing the configuration parser
parser = ConfigParser()
# Reading the config file
parser.read('config.ini')

# Load the config
print("Loading - config file ..")
CONSUMER_KEY = parser.get('app', 'CONSUMER_KEY')
CONSUMER_SECRET = parser.get('app','CONSUMER_SECRET')
ACCESS_TOKEN = parser.get('app','ACCESS_TOKEN')
ACCESS_SECRET = parser.get('app','ACCESS_SECRET')
FRIENDS_TO_PULL = parser.getint('app','FRIENDS_TO_PULL')
#We will use the getint method because we know that it is an integer
TWEETS_TO_PULL = parser.getint('app','TWEETS_TO_PULL')
MAPPED_STR = parser.get('app', 'MAPPING').replace(' ', '').split(',')

mapping = {}
for i in range(0, len(MAPPED_STR)):
	mapping[int(MAPPED_STR[i].split(':')[0])] = MAPPED_STR[i].split(':')[1]

