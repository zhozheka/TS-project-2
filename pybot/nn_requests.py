#! /usr/bin/python3
# -*- coding: utf-8 -*-

from tools import safe_call
import re
import json
import requests

from mysocketserver import send_data

class NNRequests:
    def __init__(self, group_id, token, vkapi, obj):
        self.group_id = group_id
        self.token = token
        self.object = obj
        self.api = vkapi
    def send_answer(self):
        try:
            att0 = self.object['attachments'][0]['photo']
            photo_key = self.get_photo_key(att0, 'max')
            photo_url = self.object['attachments'][0]['photo'][photo_key]
            try:
                processing_key = int(self.object['body'])
            except Exception as e:
                processing_key = 0
            photo_data = self.upload_processed_photo(photo_url, processing_key)
            photo_ident = 'photo{}_{}_{}'.format(photo_data['owner_id'], photo_data['id'], photo_data['access_key'])
        except Exception as e:
             photo_ident = None
             print(e)

        if photo_ident is not None:
            atts = [photo_ident]
        else:
            atts = []

        answer = 'Hello! Your processed image:\n{}\n'.format(self.object['body'])

        safe_call(
                self.api.messages.send,
                user_id=self.object['user_id'],
                message=answer,
                attachment=atts
            )

    def file_by_url(self, url):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            # print(type(r.content))
            return r.content

    def upload_photo(self, data):
        upload_server = safe_call(
                                self.api.photos.getMessagesUploadServer,
                                access_token=self.token
                            )
        upload_url = upload_server['upload_url']

        post_fields = {'photo': ('smth.png', data)}

        response = requests.post(upload_url, files=post_fields)
        file_data = json.loads(response.text)
        file_data['access_token'] = self.token

        photo = safe_call(self.api.photos.saveMessagesPhoto, **file_data)[0]

        return photo

    def upload_remote_photo(self, url_remote):
        remote_file = self.file_by_url(url_remote)
        return self.upload_photo(remote_file)

    def upload_processed_photo(self, url_remote, key=0):
        remote_file = self.file_by_url(url_remote)
        processed_file = send_data(bytes([key])+remote_file)
        return self.upload_photo(processed_file)

    def get_photo_key(self, data, t=''):
        re_photo = re.compile('photo_(\d+)')
        photos = []
        for key in data:
            match_photo = re_photo.match(key)
            if match_photo:
                photos.append(int(match_photo.group(1)))
        if photos == []:
            return None
        else:
            sorted_keys = sorted(photos)
            if t == 'max':
                key = sorted_keys[-1]
            elif t == 'min':
                key = sorted_keys[0]
            else:
                # key = sorted_keys[int(len(sorted_keys) / 2)]
                key = 604
            return 'photo_{}'.format(key)
