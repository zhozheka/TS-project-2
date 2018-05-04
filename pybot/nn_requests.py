#! /usr/bin/python3
# -*- coding: utf-8 -*-

from tools import safe_call

class NNRequests:
    def __init__(self, group_id, vkapi, obj):
        self.group_id = group_id
        self.object = obj
        self.api = vkapi
    def send_answer(self):
        answer = 'hello from bot! Your message was:\n{}'.format(self.object['body'])
        safe_call(
                self.api.messages.send,
                user_id=self.object['user_id'],
                message=answer
            )
