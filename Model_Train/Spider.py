'''
import os
import requests
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import base64
def download_images(image_urls, save_path, start_index=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, url in enumerate(image_urls):
        index = start_index + i
        if url is None:
            continue
        if url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url)
            with open(os.path.join(save_path, f'image_{index}.jpg'), 'wb') as f:
                f.write(response.content)
                print(f'Image {index} downloaded.')
        elif url.startswith('data:image/jpeg;base64,'):
            imgdata = base64.b64decode(url.split(',')[1])
            with open(os.path.join(save_path, f'image_{index}.jpg'), 'wb') as f:
                f.write(imgdata)
                print(f'Image {index} downloaded.')

def main():
    driver = webdriver.Chrome(r'/Users/hubertlee/Downloads/chromedriver-mac-x64/chromedriver')
    base_url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1691500693855_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=李治廷&pn={}"
    save_path = r'/Users/hubertlee/Desktop/Spider_datas'

    total_images = 0
    for page in range(10):
        print(f'Downloading page {page + 1}...')
        url = base_url.format(page * 60)
        driver.get(url)
        time.sleep(5)

        image_elements = driver.find_elements_by_tag_name('img')
        image_urls = [img.get_attribute('src') for img in image_elements]
        download_images(image_urls, save_path, start_index=total_images)

        total_images += len(image_urls)

    driver.quit()

if __name__ == '__main__':
    main()
'''

import os
import requests
from selenium import webdriver
import time
import base64

def download_images(image_urls, save_path, start_index=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, url in enumerate(image_urls):
        try:
            index = start_index + i
            if url is None:
                continue
            if url.startswith('http://') or url.startswith('https://'):
                response = requests.get(url)
                with open(os.path.join(save_path, f'image_{index}.jpg'), 'wb') as f:
                    f.write(response.content)
                    print(f'Image {index} downloaded.')
            elif url.startswith('data:image/jpeg;base64,'):
                imgdata = base64.b64decode(url.split(',')[1])
                with open(os.path.join(save_path, f'image_{index}.jpg'), 'wb') as f:
                    f.write(imgdata)
                    print(f'Image {index} downloaded.')
        except Exception as e:
            print(f'Error downloading image {index}: {e}')

def main():
    driver = webdriver.Chrome(r'/Users/hubertlee/Downloads/chromedriver-mac-x64/chromedriver')
    save_path = r'/Users/hubertlee/Desktop/Spider_datas'
    base_url = 'the url that u wanna download'
    total_images = 0
    for page in range(10):
        print(f'Downloading page {page + 1}...')
        url = base_url.format(page * 60)
        driver.get(url)
        time.sleep(5)

        image_elements = driver.find_elements_by_tag_name('img')
        image_urls = [img.get_attribute('src') for img in image_elements]
        download_images(image_urls, save_path, start_index=total_images)

        total_images += len(image_urls)

    driver.quit()

if __name__ == '__main__':
    main()



