#! /usr/bin/python3
# -*- coding: utf8 -*-

import traceback

from flask import Flask, request, json
application = Flask(__name__)

from tools import get_api, Benchmark
from nn_requests import NNRequests

app_keys = {
    'nn_bot': 'ac719611',
}

@application.route('/')
def hello():
    return 'result'

@application.route('/app/nn_bot/<path:secret>', methods=['POST'])
def c_common(secret):
    global app_keys
    token = '44469681d2d327f801b674ab6ea595b200871d33b518730b90ed99a5377b3bf8d478332bb71150e1cb699'

    rtoken = token[::-1]
    if secret != app_keys['nn_bot']:
        print('Bad agent request to {}'.format(secret))
        return 'bad agent'
    benchmark = Benchmark(request.url)
    data = json.loads(request.data)
    try:
        rtype = data['type']
        if rtype == 'confirmation':
            return app_keys['nn_bot']
            print('confirmation')
        elif rtype == 'message_new':
            vkapi = get_api(access_token=rtoken, v='5.74')
            group_id = data['group_id']
            obj = data['object']
            processor = NNRequests(group_id, rtoken, vkapi, obj)
            print('some action')
            processor.send_answer()

    except Exception as e:
        print('-----')
        print('Data:')
        print(data)
        print('Exception:')
        print(e)
        traceback.print_tb(e.__traceback__)
        print('-----')

    print(benchmark.result())
    return 'ok'

if __name__ == "__main__":
    print('bot started')
    application.run(host='0.0.0.0', port=3005)
