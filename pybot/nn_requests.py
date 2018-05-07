#! /usr/bin/python3
# -*- coding: utf-8 -*-

from tools import safe_call
import re
import json
import requests

class NNRequests:
    def __init__(self, group_id, token, vkapi, obj):
        self.group_id = group_id
        self.token = token
        self.object = obj
        self.api = vkapi
    def send_answer(self):
        try:
            att0 = self.object['attachments'][0]['photo']
            photo_key = self.get_max_photo_key(att0)
            photo_url = self.object['attachments'][0]['photo'][photo_key]
            photo_data = self.upload_remote_photo(photo_url)
            photo_ident = 'photo{}_{}_{}'.format(photo_data['owner_id'], photo_data['id'], photo_data['access_key'])
        except Exception as e:
             photo_ident = None
             print(e)

        if photo_ident is not None:
            atts = [photo_ident]
        else:
            atts = []

        answer = 'hello from bot! Your message was:\n{}\n'.format(self.object['body'])

        safe_call(
                self.api.messages.send,
                user_id=self.object['user_id'],
                message=answer,
                attachment=atts
            )

    def upload_remote_photo(self, url_remote):
        def file_by_url(url):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                # print(type(r.content))
                return r.content

        upload_server = safe_call(
                                self.api.photos.getMessagesUploadServer,
                                access_token=self.token
                            )
        upload_url = upload_server['upload_url']

        remote_file = file_by_url(url_remote)

        post_fields = {'photo': ('smth.png', remote_file)}

        response = requests.post(upload_url, files=post_fields)
        file_data = json.loads(response.text)
        file_data['access_token'] = self.token

        photo = safe_call(self.api.photos.saveMessagesPhoto, **file_data)[0]

        return photo

    def get_max_photo_key(self, data):
        re_photo = re.compile('photo_(\d+)')
        photos = []
        for key in data:
            match_photo = re_photo.match(key)
            if match_photo:
                photos.append(int(match_photo.group(1)))
        if photos == []:
            return None
        else:
            return 'photo_{}'.format(sorted(photos, reverse=True)[0])
