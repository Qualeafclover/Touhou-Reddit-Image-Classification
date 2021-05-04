import praw
import time
import requests
import numpy as np
import cv2
import urllib3
from utils import *
from configs import *
from urllib3.exceptions import InsecureRequestWarning, ProtocolError
urllib3.disable_warnings(InsecureRequestWarning)

import tensorflow as tf
thclist = THClist(all_characters, all_sisters)
model = tf.keras.models.load_model(REDDIT_MODEL_DIR)

reddit = praw.Reddit(client_id=REDDIT_ID, client_secret=REDDIT_SECRET, user_agent=f'User-Agent:{REDDIT_ID}:v0.1.818 (by /u/Qualeafclover)',
                     password=REDDIT_PASSWORD, username=REDDIT_USERNAME
                     )
reddit.validate_on_submit = True

while True:
    for mention in reddit.inbox.mentions(limit=None):
        if ONLY_ME:
            if mention.author != CREATOR:
                continue

        # Checks already replied
        if mention not in get_txt_file(REPLIED_TXT):
            append_txt_file(str(mention)+'\n', REPLIED_TXT)
            print(f'Seeing {mention} for the first time...')
        else: continue

        # Checks subreddit
        if (mention.subreddit == SUBREDDIT) or ALL_SUBREDDIT:
            # Checks image post
            url = mention.submission.url
            if url.startswith('https://i.redd.it/'):
                # Download image
                while True:
                    try:
                        res = requests.get(url=url, verify=False, stream=True)
                        rawdata = res.raw.read()
                        break
                    except (ConnectionError, ConnectionAbortedError, ProtocolError):
                        time.sleep(1)
                nparr = np.frombuffer(rawdata, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.resize(image, RESIZE)
                image = np.expand_dims(image, 0)
                image = image.astype(np.float32)
                image = image/255

                pred = model.predict(image)
                pred = thclist.one_hot_decode(pred, top=3, round_num=3, english=True)[0]

                reply = 'Here is what I think it is...\n\n'
                for character in pred:
                    reply += f'{character}: {round(pred[character]*100, 3)}%\n\n'
                reply += '''Kappa kappa I am a bot made by Nitori~'''

                print(pred)
                print(reply + '\n')
                mention.reply(reply)
    # Yeah take a break
    time.sleep(5)