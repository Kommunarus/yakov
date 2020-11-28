from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from os import makedirs as make_dirs
from os.path import join as join_path
from os.path import abspath
from os.path import exists as file_exist
import subprocess

import json
import uuid
import csv

HTML_TEMPLATE_DIR = './templates'
STATIC_DIR = './static'
TEMP_DIR = STATIC_DIR+'/temp'

app = Flask(__name__, static_url_path='', template_folder=HTML_TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['DEBUG'] = True
CORS(app)


@app.route('/', methods=['GET'])
def main_page():
    return render_template('index.html')


@app.route('/transform', methods=['POST'])
def transform():
    request_data = request.get_json()
    input_text = request_data.get('inputText', None)
    if input_text:
        req_uuid = str(uuid.uuid1())
        # req_uuid = "7ec3d6ec-30c1-11eb-a8b1-48ba4e478f7f"
        temp_dir = store_to_temp_dir(input_text, req_uuid) + '/'
        # Вызов функции тут

        command = '/home/alex/anaconda3/envs/coref/bin/python'
        path2script = '/home/alex/PycharmProjects/coref/pipline.py'
        cmd = [command, path2script] + ['--outdir'] + [temp_dir]
        subprocess.check_output(cmd, universal_newlines=True)

        response_object = get_result(req_uuid)
        return json.dumps(response_object)

    response_object = {
        'outputText': 'Заглушка'
    }
    return json.dumps(response_object)


def store_to_temp_dir(text, req_id):
    temp_dir = abspath(join_path(TEMP_DIR, req_id))
    make_dirs(temp_dir, exist_ok=True)
    output_file = join_path(temp_dir, 'input.txt')
    with open(output_file, encoding='utf-8', mode='w') as f:
        f.write(text)
    return temp_dir


def get_result(req_id):
    result = {
        'inputText': '',
        'outputText': '',
        'clusterText': '',
        'replacements': [],
        'requestId': req_id
    }
    result_dir = abspath(join_path(TEMP_DIR, req_id))

    input_file = join_path(result_dir, 'input.txt')
    if file_exist(input_file):
        result['inputText'] = read_text_file_content(input_file)

    output_file = join_path(result_dir, 'output.txt')
    if file_exist(output_file):
        result['outputText'] = read_text_file_content(output_file)

    cluster_file = join_path(result_dir, 'cluster.txt')
    if file_exist(cluster_file):
        result['clusterText'] = read_text_file_content(cluster_file)

    replacement_file = join_path(result_dir, 'output.csv')
    if file_exist(replacement_file):
        result['replacements'] = read_replacement_file(replacement_file)

    return result


def read_text_file_content(file_path):
    file_content = ''
    with open(file_path, encoding='utf-8', mode='r') as f:
        file_content = f.read()
    return file_content


def read_replacement_file(file_path):
    replacements = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            replacements.append(row)
    return replacements


if __name__ == '__main__':
    app.run()
