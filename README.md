# yakov
Задача.
Учитывая специфику деятельности определенных служб МВД России, при подготовке документов требуется преобразование в тексте лица повествования от первого лица в третье с учетом рода. Например, фраза в исходном тексте «Я увидел, что Иванов пошёл ко мне» в итоговом тексте должна быть преобразована в «Он увидел, что Иванов пошёл к нему». Разработанное программное решение позволит в автоматическом режиме проводить процесс конвертации лица повествования, что позволит сотрудникам уделить больше времени на иные аспекты служебной деятельности. Кейс подготовлен Департаментом информационных технологий, связи и защиты информации МВД России

Решение.
Основные идеи, лежащии в решении: 
1. Лемматизация всех слов текста. При этом местоимения первого лица переводятся в слово “я”. это будут наши главные кандидаты на изменения своего вида.

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/1.png)

2. поиск зависимых (главных) слов у найденных местоимений (используется библиотека udpipe из R).

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/2.png)

3. если главное слово найдено и у него хороший score (вероятность, что слово в словаре и его форма определена правильно), то определяем его род, число, морфологию. И запоминаем.

4. если score низкий (это означает что слово не в словаре или оно написано с опечаткой), то через яндекс спеллер проверяем его орфографию. Если слово по мнению спеллера корректное, то оставляем его в первоначальном виде. Если слово не корректное, то используем исправленное слово, а если у спеллера нет исправленного слова, то ищем ближайщее похожее сравнивая эмбединги слов (fasttext).

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/3.png)

5. полученное слово из п.4 меняется на третье лицо через лексемы библиотеки.

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/22.png)

6. в цикле меняем слова местоимений. Род определяется как самый частый род у главных слов.
		'я': ('он','она'),
		'меня': ('его','её'),
		'мне': ('ему','ей'),
		'мной': ('им','ею'),
		'мною': ('им','ею'),
		….
если местоимение перед собой имеет предлог, то добавляем букву “н”.


Это основной алгоритм. Но он страдает одним недостатком. Он может делать замену там, где она не нужна. Например, если в тексте встречается прямая речь другого человека. Пример:
“Я спросил его, что он делал. Он сказал: “я ждал тебя””.
Второе “я” нельзя менять. Поэтому нам нужен способ определения только таких местоимений, которые относятся к главному лицу. В nlp это называется “coreference resolution”.

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/4.png)

Такая задача решается через сложную систему правил, например алгоритм Хоббса.  Но таких привил должно быть много, их развитие и поддержка очень затратна и по деньгам и по времени. поэтому в 2017 году была представлена end-to-end сеть, которая была призвана заменит сложный свод правил на одну нейросеть. 


![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/5.png)

Сетки подобной структуры, обученные на английском датасете (2500 текстов) получает  точность по F1 = 80, что является очень высоким результатом.

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/6.png)

Мы не нашли готовой обученной сети для русского датасета, поэтому решили ее  модернизировать и перевести на свой датасет и дообучить на текстах из интернета.

Привет- ,- меня(0) зовут- Михаил- ,- мне(0) 30- лет- и- я(0) живу- в- Москве- .- Я(0) хочу(0) рассказать- про- волка- .- Он(1) был- злой- и- темный- .- Его(1) я(0) постараюсь- забыть- .-

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/21.png)

на рисунке видно, что сеть выделила два кластера слов: все те, что относятся к “я” и те, что относятся к “он” (к волку). И это сделала сеть, обученная на 25 небольших размеченных текстах. Предполагается, что все местоимения “я”, которые сказал собеседник главного повествователя, будут выделяться во втором (второстепенный) кластере (при условии хорошо обученной сетки).

Итак. Наше решение предполагает: обучение сети  coreference resolution на большом датасете объяснительных. Первоначальный запуск сети на обрабатываемом тексте, получение кластера местоимений, в котором есть больше слов “я”. Менять на 3 лицо только слова из кластера. На небольших текстах работа сети – несколько секунд.


Результаты.
1. сделан макрос для либраофиса.

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/31.png)


![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/33.png)


2. сделан чат бот телеграмма для удаленной работы. Нужно передать ему текстовый файл. В ответ вернется обработанный.
@mksk0m_bot. 

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/30.png)

3. Сайт http://yakov.mkskom.ru/

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/32.png)


ПС.
Для работы автокорректора опечаток можно использовать онлайн проверк от яндекса или использовать ближайших соседей по векторам fasttext (скачать можно отсюда https://rusvectores.org/)
Для обучения сети для разрешения кореференций нужно дополнительно разметить массив документов. 

![рисунок](https://github.com/Kommunarus/yakov/blob/master/img/40.png)


Основные файлы.

coreference_resolution/src/coref.py - обучение сети разрешения кореференции \n
bot/bot.py - запуск телеграмм бота \n
flask/ - сайт \n
cluster.py - поиск кластеров в тексте
pipline.py- скрипт замены лица
mvd2020.py - макрос для либрофиса
