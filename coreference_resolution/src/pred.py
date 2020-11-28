from coreference_resolution.src.coref import CorefScore, Trainer
from coreference_resolution.src.loader import load_glove, load_dataset
from subprocess import Popen, PIPE
import collections
import os
import codecs
import subprocess
import re

rx_metric = re.compile('METRIC ([a-z]+):')
rx_score = re.compile('([A-Za-z\- ]+): Recall:.* ([0-9\.]+)%\tPrecision:.* ([0-9\.]+)%\tF1:.* ([0-9\.]+)%')

eval_script='../src/eval/scorer.pl'
savemodel = '2020-11-19 08:02:33.645418.pth'

_, val_corpus, _ = load_dataset(t=False, v=True, te=False)
GLOVE = load_glove()

model = CorefScore(embeds_dim=350, hidden_dim=200, GLOVE = GLOVE)

trainer = Trainer(model, [], val_corpus, steps=100)
trainer.load_model(savemodel)

# val_corpus = [doc for doc in val_corpus if doc.sents]

predicted_docs = [trainer.predict(doc) for doc in val_corpus]

val_corpus.docs = predicted_docs

# Output results
golds_file, preds_file = trainer.to_conll(val_corpus, eval_script)
metrics=('muc', 'bcub', 'ceafm')
# metrics = ('all')
results = ''

for metric in metrics:
    # Run perl script
    scorer_params = ['perl',
                     eval_script,
                     metric,
                     golds_file,
                     preds_file,
                     'none']

    # print('Running Perl evaluation script...')
    output = subprocess.check_output(scorer_params).decode('utf-8').split('\n')
    scores = collections.defaultdict(dict)
    for line in output:
        line = line.strip('\r\n')
        m = rx_metric.match(line)
        if m:
            metric = m.group(1)
        m = rx_score.match(line)
        if m:
            scores[metric][m.group(1)] = (m.group(2), m.group(3), m.group(4))
    results += str(scores)+'\n'

print(results)