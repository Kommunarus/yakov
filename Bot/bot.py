import requests
import datetime
import  subprocess

token = ''
path = '/home/alex/PycharmProjects/coref/Bot/'

class BotHandler:

    def __init__(self, token):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)

    def get_updates(self, offset=None, timeout=30):
        method = 'getUpdates'
        params = {'timeout': timeout, 'offset': offset}
        resp = requests.get(self.api_url + method, params)
        result_json = resp.json()['result']
        return result_json

    def send_message(self, chat_id, text):
        params = {'chat_id': chat_id, 'text': text}
        method = 'sendMessage'
        resp = requests.post(self.api_url + method, params)
        return resp

    def send_document(self, chat_id, doc):
        # params = {'chat_id': chat_id, 'document': open(doc, 'rb')}
        # method = 'sendDocument'
        # resp = requests.post(self.api_url + method, params)
        files = {'document': open('output.txt', 'rb')}
        resp = requests.post("https://api.telegram.org/bot"+token+"/sendDocument?chat_id=" + str(chat_id), files=files)
        return resp

    def get_last_update(self):
        get_result = self.get_updates()

        if len(get_result) > 0:
            last_update = get_result[-1]
        else:
            last_update = None
        return last_update

    def startWorkNN(self, file_name, file_id, chat_id):
        params = {'file_id': file_id}
        method = 'getFile'
        resp = requests.post(self.api_url + method, params)
        if resp.status_code == 200:
            file_path = resp.json()['result']['file_path']
            r = requests.get("https://api.telegram.org/file/bot" + token + "/" + file_path, allow_redirects=True)
            open('input.txt', 'wb').write(r.content)
            command = '/home/alex/anaconda3/envs/coref/bin/python'
            path2script = '/home/alex/PycharmProjects/coref/pipline.py'
            cmd = [command, path2script] + ['--outdir'] + [path]
            subprocess.check_output(cmd, universal_newlines=True)

            self.send_document(chat_id, 'output.txt')
            self.send_message(chat_id, 'Вам выслан результат обработки.')




greet_bot = BotHandler(token)
greetings = ('здравствуй', 'привет', 'ку', 'здорово')
now = datetime.datetime.now()


def main():
    new_offset = None
    today = now.day
    hour = now.hour
    last_update_id = 0

    while True:
        greet_bot.get_updates(new_offset)

        last_update = greet_bot.get_last_update()
        last_chat_text = ''
        file_id = ''
        if last_update != None:
            last_update_id = last_update['update_id']
            last_chat_id = last_update['message']['chat']['id']
            if 'text' in last_update['message']:
                last_chat_text = last_update['message']['text']
            if  'document' in last_update['message']:

                file = last_update['message']['document']
                file_name = file['file_name']
                file_id = file['file_id']
                greet_bot.send_message(last_chat_id, 'Мы получили Ваш файл. Подождите несколько секунд.')
                greet_bot.startWorkNN(file_name, file_id, last_chat_id)
            last_chat_name = last_update['message']['chat']['first_name']

            if last_chat_text.lower() in greetings and today == now.day and 6 <= hour < 12:
                greet_bot.send_message(last_chat_id, 'Доброе утро, {}'.format(last_chat_name))
                today += 1

            elif last_chat_text.lower() in greetings and today == now.day and 12 <= hour < 17:
                greet_bot.send_message(last_chat_id, 'Добрый день, {}'.format(last_chat_name))
                today += 1

            elif last_chat_text.lower() in greetings and today == now.day and 17 <= hour < 23:
                greet_bot.send_message(last_chat_id, 'Добрый вечер, {}'.format(last_chat_name))
                today += 1

        new_offset = last_update_id + 1

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
