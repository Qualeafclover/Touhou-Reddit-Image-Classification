from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from urllib3.exceptions import InsecureRequestWarning, ProtocolError
from requests.utils import unquote
import urllib3
import requests

import numpy as np
import cv2
import os
import sys
import time
import copy
from datetime import timedelta
from utils import *
from configs import *
import random

urllib3.disable_warnings(InsecureRequestWarning)

class PixivDataset(object):
    def __init__(self, batches=BATCHES, preload=False):
        self.THClist = THClist(all_characters, all_sisters)

        self.preloaded = preload
        images = list(self.load_images())
        random.seed(SEED)
        random.shuffle(images)
        X, y = self.load_data(images)
        test_datasets = round(y.shape[0]*TEST_PERCENTAGE)

        test_X,  test_y  = X[:test_datasets], y[:test_datasets]
        train_X, train_y = X[test_datasets:], y[test_datasets:]

        self.train_ds = Dataset(train_X, train_y, test=False, batches=batches, preloaded=preload)
        self.test_ds  = Dataset(test_X,  test_y,  test=True,  batches=batches, preloaded=preload)

    def load_data(self, images):
        y = []
        image_dict = {
            line.split()[0].split('/')[-1].split('_')[0]+'.jpg': line.split()[1:] for line in get_txt_file(PICTURE_INFO_TXT)
        }
        rm_img = []
        for image in images:
            value = image_dict[image.split('\\')[-1]]
            value = self.THClist.one_hot_encode(value)
            if True not in value:
                rm_img.append(image)
                continue
            y.append(value)
        y = np.array(y, dtype=np.float32)

        if self.preloaded:
            X = np.stack(list(cv2.resize(spl_imread(os.path.join(IMAGE_DIR, image)), RESIZE) for image in images if (image not in rm_img)))
        else:
            X = np.array([os.path.join(IMAGE_DIR, image) for image in images if (image not in rm_img)])

        return X, y

    def load_images(self):
        for imdir in os.listdir(IMAGE_DIR):
            image = os.path.join(IMAGE_DIR, imdir)
            yield image

    def one_hot_decode(self, *args, **kwargs):
        return self.THClist.one_hot_decode(*args, **kwargs)

class Dataset(object):
    def __init__(self, X, y, test=False, batches=BATCHES, preloaded=False):
        self.num_samples = y.shape[0]
        self.batches = batches
        self.num_batches = int(np.ceil(y.shape[0] / self.batches))
        self.test = test

        self.X, self.y = X, y
        self.preloaded = preloaded

    def load_image(self, path):
        image = spl_imread(path)
        image = cv2.resize(image, RESIZE)
        return image

    def augmentate(self, image):
        if self.test: return image
        # RANDOM FLIPPING #
        if random.random() < 0.5:
            image = image[:, ::-1, :]

        # RANDOM BRIGHTNESS #
        if random.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.int16)
            hsv[:, :, 2] += random.randint(-50, 50)
            hsv = np.clip(hsv, 0, 255)
            hsv = hsv.astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # RANDOM SATURATION #
        if random.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.int16)
            hsv[:, :, 1] += random.randint(-30, 30)
            hsv = np.clip(hsv, 0, 255)
            hsv = hsv.astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # RANDOM HUE #
        if random.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.int16)
            hsv[:, :, 0] += random.randint(-10, 10)
            hsv = np.clip(hsv, 0, 255)
            hsv = hsv.astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # COLOR SHIFT #
        if random.random() < 0.5:
            rgb = image.astype(np.int16)
            rgb[:, :, 0] += random.randint(-5, 5)
            rgb[:, :, 1] += random.randint(-5, 5)
            rgb[:, :, 2] += random.randint(-5, 5)
            rgb = np.clip(rgb, 0, 255)
            image = rgb.astype(np.uint8)

        # RANDOM CROP #
        if random.random() < 0.5:
            hmin, hmax, wmin, wmax = 0, image.shape[0], 0, image.shape[1]
            hmin += random.randint(0, 20)
            hmax += random.randint(-20, 0)
            wmin += random.randint(0, 20)
            wmax += random.randint(-20, 0)
            image = image[hmin:hmax, wmin:wmax]
            image = cv2.resize(image, RESIZE)

        return image

    def __iter__(self):
        self.batch_count = 0
        self.X, self.y = unison_shuffled_copies(self.X, self.y)
        return self

    def __next__(self):
        if self.batch_count < self.num_batches:
            X = self.X[self.batches*self.batch_count:self.batches*(self.batch_count+1)]
            y = self.y[self.batches*self.batch_count:self.batches*(self.batch_count+1)]

            if not self.preloaded:
                X = np.stack(list(self.augmentate(self.load_image(X1)) for X1 in X))
            else:
                X = np.stack(list(self.augmentate(X1) for X1 in X))

            X = X.astype(np.float32)
            X = X/255

            y = y.astype(np.float32)

            self.batch_count += 1
            return X, y
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batchs

class PixivGrabber(object):
    class CaptchaException(Exception):
        pass
    class InsufficientDownloadsException(Exception):
        pass

    def __init__(self, init_driver=True, last_page=1):
        if init_driver:
            self.init_driver()
            if LOGIN: self.login()

        self.seen_txt         = URL_SEEN_TXT
        self.uninspected_urls = []

        self.pic_info_txt   = PICTURE_INFO_TXT
        self.training_queue = []

        self.trained_txt = TRAINED_PICTURE_TXT

        self.last_page = last_page

        if RESET_FILES: self.reset_files()

        self.THClist = THClist(all_characters, all_sisters)

    def get_urls(self, tag=TAG, quiet=False):
        driver = self.driver
        driver.get(self.webpage.format(tag, self.last_page))

        if not quiet:
            print('Scanning page number {}'.format(self.last_page))
        while True:
            try:
                driver.find_element_by_css_selector("a[class='rp5asc-16 kdmVAX sc-AxjAm MksUu']")
                image_container = driver.find_elements_by_css_selector("a[class='rp5asc-16 kdmVAX sc-AxjAm MksUu']")[4:]
                if len(image_container) == 0: raise NoSuchElementException
                break
            except NoSuchElementException: pass
        if not quiet:
            print('Obtained {} urls'.format(len(image_container)))

        album_url, seen_url = 0, 0
        for image_item in image_container:
            try:
                image_item.find_element_by_css_selector("div[class='sc-1mr081w-0 gWvsci']")
                album_url += 1
            except NoSuchElementException:
                url = image_item.get_attribute('href')
                if url not in self.uninspected_urls + get_txt_file(self.seen_txt):
                    self.uninspected_urls.append(url)
                    append_txt_file(url+'\n', self.seen_txt)
                else:
                    seen_url += 1
        if not quiet:
            if album_url != 0:
                print('Dropped {} album urls'.format(album_url))
            if seen_url != 0:
                print('Dropped {} seen urls'.format(seen_url))
            print('Obtained {} new urls'.format(len(image_container)-album_url-seen_url))
            print()
        self.last_page += 1

    def inspect_urls(self, start=0, quiet=False):
        total_uninspected = len(self.uninspected_urls[start:])
        uninspected_copy = copy.copy(self.uninspected_urls[start:])

        driver = self.driver
        tpc = time.perf_counter
        t = tpc()

        skips = 0
        for url in self.uninspected_urls[start:]:
            if url.split('/')[-1] in [line[0].split('/')[-1].split('_')[0] for line in self.training_queue]:
                if not quiet: print('Image already inspected, skipping...')
                skips += 1
                continue

            if not quiet:
                url_num = uninspected_copy.index(url)
                prefix = f'Inspecting image: {url_num+1}/{total_uninspected}'
                taken, left = tpc() - t, total_uninspected - url_num
                try: est = str(timedelta(seconds=round(taken / (url_num-skips) * left)))
                except ZeroDivisionError: est = '?:??:??'
                print('{}{}Estimated time left: {}'.format(prefix, (35-len(prefix))*' ', est))

            driver.get(url)
            ctn = False
            timeout_counter = tpc()
            while True:
                if tpc() - timeout_counter > TIMEOUT:
                    ctn = True
                    break
                try:
                    try:
                        driver.find_element_by_css_selector("svg[class='sc-1w3e579-0 bmyrlo']")
                        if not quiet: print('Motion picture detected, skipping...')
                        ctn = True
                        break
                    except NoSuchElementException: pass
                    try:
                        driver.find_element_by_css_selector("img[src='https://d.pixiv.org/file?format=default&creative_id=20685']")
                        if not quiet: print('Deleted picture detected, skipping...')
                        ctn = True
                        break
                    except NoSuchElementException: pass
                    driver.find_element_by_css_selector("a[class='gtm-new-work-tag-event-click']")
                    tag_container = driver.find_elements_by_css_selector("a[class='gtm-new-work-tag-event-click']")
                    tag_container = [unquote(tag.get_attribute('href').split('/')[5]) for tag in tag_container]
                    try:
                        image_url = driver.find_element_by_css_selector("img[class='sc-1qpw8k9-1 dpYYLs']").get_attribute('src')
                    except NoSuchElementException:
                        image_url = driver.find_element_by_css_selector("img[class='sc-3xfm45-0 bqeYLx']").get_attribute('src')
                    driver.execute_script("window.stop();")
                    break
                except NoSuchElementException:
                    pass
            if ctn:
                driver.execute_script("window.stop();")
                self.uninspected_urls.remove(url)
                continue
            tag_container.sort()

            s = image_url + ' ' + ' '.join(tag_container) + '\n'

            if s[:-1] in get_txt_file(self.pic_info_txt):
                if not quiet: print('Image already found in text file, skipping...')
            else:
                try:
                    append_txt_file(s, self.pic_info_txt)
                    self.training_queue.append((image_url, tag_container))
                except UnicodeEncodeError:
                    if not quiet: print('Non-supported character detected, skipping...')
        if not quiet: print()

    def download_urls(self, batches=BATCHES, get_more=False, quiet=False):
        if len(self.training_queue) < batches:
            if get_more:
                if not quiet:
                    print('Insufficient urls, getting more')
                    print()
                while len(self.uninspected_urls) + len(self.training_queue) < batches:
                    self.get_urls(quiet=quiet)
                while len(self.training_queue) < batches:
                    self.inspect_urls(quiet=quiet)
            else:
                raise self.InsufficientDownloadsException
        headers = {'referer': 'https://www.pixiv.net/'}
        batch, self.training_queue = self.training_queue[:BATCHES], self.training_queue[BATCHES:]

        batch_copy = copy.copy(batch)
        tpc = time.perf_counter
        t = tpc()
        for img_tags in batch[:]:
            img, tags = img_tags
            if not quiet:
                item_num = batch_copy.index(img_tags)
                prefix = f'Downloading image: {item_num+1}/{batches}'
                taken, left = tpc() - t, batches - item_num
                try: est = str(timedelta(seconds=round(taken / item_num * left)))
                except ZeroDivisionError: est = '?:??:??'
                print('{}{}Estimated time left: {}'.format(prefix, (35-len(prefix))*' ', est))

            while True:
                try:
                    res = requests.get(img, headers=headers, verify=False, stream=True)
                    rawdata = res.raw.read()
                    break
                except (ConnectionError, ConnectionAbortedError, ProtocolError):
                    if not quiet: print('Error')
                    time.sleep(1)

            nparr = np.frombuffer(rawdata, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None: continue
            yield [image, self.THClist.get_characters(tags)]
        if not quiet: print()

    def download_all(self, start=0, quiet=False):
        headers = {'referer': 'https://www.pixiv.net/'}
        tpc = time.perf_counter
        t = tpc()

        skips = 0
        total_queue = len(self.training_queue[start:])
        queue_copy = copy.copy(self.training_queue[start:])
        for img_tags in self.training_queue[start:]:
            img, tags = img_tags
            filename = img.split('/')[-1].split('_')[0] + '.jpg'

            if filename in os.listdir(IMAGE_DIR):
                if not quiet: print('Image already downloaded, skipping...')
                skips += 1
                continue

            if not quiet:
                item_num = queue_copy.index(img_tags)
                prefix = f'Downloading image: {item_num+1}/{total_queue}'
                taken, left = tpc() - t, total_queue - item_num
                try: est = str(timedelta(seconds=round(taken / (item_num-skips) * left)))
                except ZeroDivisionError: est = '?:??:??'
                print('{}{}Estimated time left: {}'.format(prefix, (35-len(prefix))*' ', est))

            while True:
                try:
                    res = requests.get(img, headers=headers, verify=False, stream=True)
                    rawdata = res.raw.read()
                    break
                except (ConnectionError, ConnectionAbortedError, ProtocolError):
                    if not quiet: print('Error')
                    time.sleep(1)

            nparr = np.frombuffer(rawdata, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None: continue

            h, w, c = image.shape
            if h > w: image = cv2.resize(image, (round(w*(MIN_DOWNLOAD_AXIS/h)), MIN_DOWNLOAD_AXIS))
            else: image = cv2.resize(image, (MIN_DOWNLOAD_AXIS, round(h*(MIN_DOWNLOAD_AXIS/w))))

            spl_imwrite(os.path.join(IMAGE_DIR, filename), image)

    def load_seen(self, replace=False, quiet=False):
        seen_lines = get_txt_file(self.seen_txt)
        if replace: self.uninspected_urls = seen_lines
        else: self.uninspected_urls += seen_lines

        if not quiet:
            print(f'Loaded {len(seen_lines)} new urls')
            if not replace:
                print(f'New total seen urls: {len(self.uninspected_urls)}')
            print()

    def load_info(self, replace=False, quiet=False):
        info_lines = get_txt_file(self.pic_info_txt)
        info_lines = [(s.split()[0], s.split()[1:]) for s in info_lines]
        if replace: self.training_queue = info_lines
        else: self.training_queue += info_lines

        if not quiet:
            print(f'Loaded {len(info_lines)} new urls')
            if not replace:
                print(f'New total seen urls: {len(self.training_queue)}')
            print()

    def init_driver(self):
        DesiredCapabilities.CHROME['pageLoadStrategy'] = 'none'
        self.driver  = webdriver.Chrome(CHROMEDRIVER, options=webdriver.ChromeOptions())
        self.webpage = 'https://www.pixiv.net/en/tags/{}/illustrations?p={}'

    def login(self):
        driver = self.driver
        tpc = time.perf_counter
        driver.get('https://accounts.pixiv.net/login')
        while True:
            try:
                username_input = driver.find_element_by_xpath("/html/body/div[@class='signup-form ']/div[@id='container-login']/div/form/div[@class='input-field-group']/div[1]/input")
                password_input = driver.find_element_by_xpath("/html/body/div[@class='signup-form ']/div[@id='container-login']/div/form/div[@class='input-field-group']/div[2]/input")
                submit_buttom = driver.find_element_by_xpath("/html/body/div[@class='signup-form ']/div[@id='container-login']/div/form/button")
                break
            except NoSuchElementException: pass
        username_input.send_keys(EMAIL)
        password_input.send_keys(PASSWORD)
        submit_buttom.submit()

        t = tpc()
        while driver.current_url == 'https://accounts.pixiv.net/login':
            if tpc() - t > TIMEOUT:
                driver.quit()
                raise self.CaptchaException

    def reset_files(self):
        with open(self.trained_txt, 'w') as f: pass

        # This took hours to do okay, please don't delete these files...
        # with open(self.pic_info_txt, 'w') as f: pass
        # with open(self.seen_txt, 'w') as f: pass

    def tag_stats(self, use_thc=True, ignore_unknown=True,
                  ちゃん抜き=True, hash抜き=True, ゆっくり抜き=True):
        if use_thc:
            tags = [self.THClist.get_characters(tags, ignore_unknown, ちゃん抜き, hash抜き, ゆっくり抜き)
                    for url, tags in self.training_queue]
        else:
            tags = [tag for url, tag in list(self.training_queue)]
        tags = [item for sublist in tags for item in sublist]
        tag_set = list(set(tags))

        counter = [(tags.count(tag), tag) for tag in tag_set]
        counter.sort()
        counter.reverse()
        for count, tag in counter:
            print(f'{tag}: {count}')

    def __iter__(self):
        self.epoch_count = 0
        return self

    def __next__(self):
        self.epoch_count += 1
        if self.epoch_count > EPOCHS:
            raise StopIteration
        else:
            dataset = list(self.download_urls(quiet=QUIET, get_more=True))
            X = [X for X, y in dataset]
            X = [cv2.resize(img, RESIZE) for img in X]
            X = np.array(X, dtype=np.float32)
            X = X/255

            y = [y for X, y in dataset]
            y = np.array([self.THClist.one_hot_encode(tags) for tags in y], dtype=np.float32)

            return X, y

if __name__ == '__main__':
    LOGIN = True
    RESET_FILES = False
    grabber = PixivGrabber(last_page=800, init_driver=False)
    grabber.load_seen()
    grabber.load_info()

    # for n in range(200):
    #     grabber.get_urls()

    # grabber.inspect_urls(start=23000)

    # grabber.download_all(start=30305+4600)

    # grabber.inspect_urls(start=27000+5100)
    # grabber.download_all()

    grabber.tag_stats()
