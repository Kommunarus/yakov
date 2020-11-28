import uno
import  subprocess
import random
import glob
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
import csv

path = '/home/alex/PycharmProjects/coref/outdataoffice/'


def conver_1per_to_3per():
    doc = XSCRIPTCONTEXT.getDocument()
    text = doc.getText()  # com.sun.star.text.Text

    oldtext = text.getString()
    with open(path+'input.txt', 'w') as f:
        f.write(oldtext)


    command = '/home/alex/anaconda3/envs/coref/bin/python'
    path2script = '/home/alex/PycharmProjects/coref/pipline.py'
    cmd = [command, path2script]+['--outdir']+[path]
    out = subprocess.check_output(cmd, universal_newlines=True)

    with open(path+'output.txt', 'r') as f:
        newtext = f.read()
    text.setString(newtext)

    with open(path+'coordinat2.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            cursor = text.createTextCursor()
            cursor.gotoStart(False)
            cursor.goRight(row[1], False)
            cursor.goRight(row[2], True)
            cursor.setPropertyValue("CharColor", 200*200)


def conver_1per_to_3per_comment():
    doc = XSCRIPTCONTEXT.getDocument()
    text = doc.getText()  # com.sun.star.text.Text
    cursor = text.createTextCursor()

    oldtext = text.getString()
    with open(path+'input.txt', 'w') as f:
        f.write(oldtext)


    command = '/home/alex/anaconda3/envs/coref/bin/python'
    path2script = '/home/alex/PycharmProjects/coref/pipline.py'
    cmd = [command, path2script]+['--outdir']+[path]
    out = subprocess.check_output(cmd, universal_newlines=True)

    with open(path+'output2.txt', 'r') as f:
        newtext = f.read()
    text.setString(newtext)

    with open(path+'coordinat.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            cursor.gotoStart(False)
            cursor.goRight(row[2], False)
            cursor.goRight(len(row[1].split('%')[2]), True)
            Annotation = doc.createInstance("com.sun.star.text.textfield.Annotation")
            Annotation.Content = row[1].split('%')[1]
            text.insertTextContent(cursor, Annotation, 0)

def show_clusters():
    doc = XSCRIPTCONTEXT.getDocument()
    text = doc.Text

    text = doc.getText()  # com.sun.star.text.Text

    oldtext = text.getString()
    with open(path+'input.txt', 'w') as f:
        f.write(oldtext)


    command = '/home/alex/anaconda3/envs/coref/bin/python'
    path2script = '/home/alex/PycharmProjects/coref/cluster.py'
    cmd = [command, path2script]+['--outdir']+[path]
    out = subprocess.check_output(cmd, universal_newlines=True)

    with open(path+'cluster.txt', 'r') as f:
        line1 = f.readline()
        line2 = f.readline()
        line3 = f.readline()
    tokens   = line1.strip().split('%')
    tags   = line2.strip().split('%')
    num_clusters = int(line3)

    # rColore = []
    # for i in range(num_clusters):
    #     rColore.append(random.sample(list(range(255)), 3))
    rColore = random.sample(list(range(250**3)),num_clusters)
    # flog = open('/home/alex/PycharmProjects/coref/log.txt', 'w')
    # newtext = ' '.join(['{}{}'.format(x, y) for (x, y) in zip(tokens, tags)])
    offset = 0
    num = 0
    for i in range(len(tokens)):
        num = oldtext.find(tokens[i], offset)
        if tags[i] != '-1':
            cclas = rColore[int(tags[i])]
        else:
            cclas = 0
        print('{} {} {}'.format(tokens[i], tags[i], cclas))

        cursor = text.createTextCursor()
        cursor.gotoStart(False)
        cursor.goRight(num, False)
        cursor.goRight(len(tokens[i]), True)
        cursor.setPropertyValue("CharColor", cclas)
            # cursor.setPropertyValue('CharShadowed', True)

        offset = num + len(tokens[i])
    cursor.gotoEnd(False)

    img = doc.createInstance('com.sun.star.text.TextGraphicObject')
    file_list = glob.glob(path+'*.png')

    img.GraphicURL = 'file://' + file_list[0]
    text.insertTextContent(cursor, img, False)
    text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)
    # text.setString(newtext+'\n'+oldtext)


    return

