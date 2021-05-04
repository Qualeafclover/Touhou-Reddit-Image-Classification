'''
Please edit the file directories accordingly.
Info such as passwords and usernames have been omitted, change accordingly.
'''

#===================#
# DIRECTORY CONFIGS #
#===================#
RESET_FILES         = False
PICTURE_INFO_TXT    = 'C:/Users/quale/Desktop/THC_rework/data/pic_info.txt'
TRAINED_PICTURE_TXT = 'C:/Users/quale/Desktop/THC_rework/data/trained.txt'
URL_SEEN_TXT        = 'C:/Users/quale/Desktop/THC_rework/data/urls_seen.txt'
CHROMEDRIVER        = 'C:/Users/quale/Desktop/THC_rework/chromedriver.exe'
IMAGE_DIR           = 'C:/Users/quale/Desktop/THC_rework/images'

#===============#
# PIXIV CONFIGS #
#===============#
LOGIN    = False
EMAIL    = ''
PASSWORD = ''
TAG      = '東方Project'
TIMEOUT  = 60.0
QUIET    = False

MIN_DOWNLOAD_AXIS = 600

#=============#
# CNN CONFIGS #
#=============#

SEED = 42

RESIZE = (299, 299)

TEST_PERCENTAGE = 0.10

BATCHES = 16
EPOCHS  = 15

DELETE_LOGS     = True
USE_TENSORBOARD = True
FROM_CHECKPOINT = False
LEARNING_RATE   = 0.00001

#====================#
# MODEL SAVE CONFIGS #
#====================#

MODEL_SAVE_DIR   = 'C:/Users/quale/Desktop/THC_rework/models/model({}_{}_{}).tf'
MODEL_LOAD_DIR   = 'C:/Users/quale/Desktop/THC_rework/models/model(2_1.146_0.818).tf'
LOAD_MODEL = True

#================#
# REDDIT CONFIGS #
#================#

REDDIT_MODEL_DIR = 'C:/Users/quale/Desktop/THC_rework/reddit/model.tf'

REDDIT_USERNAME = 'Nitori-bot'
REDDIT_PASSWORD = ''
REDDIT_SECRET = ''
REDDIT_ID = ''

CREATOR = 'Qualeafclover'
SUBREDDIT = 'touhou'
REPLIED_TXT = 'C:/Users/quale/Desktop/THC_rework/reddit/replied.txt'

ALL_SUBREDDIT = True
ONLY_ME = False
