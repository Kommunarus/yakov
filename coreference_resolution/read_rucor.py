from rucoref.anaphoralib.corpora import rueval
from rucoref.anaphoralib.tagsets import multeast
from rucoref.anaphoralib.tagsets.utils import same_grammemmes
from rucoref.anaphoralib.experiments import mentionpair
from rucoref.anaphoralib.experiments import coref_utils
from rucoref.anaphoralib import utils
from rucoref.anaphoralib.experiments import utils as exp_utils
import random

random.seed(2020)

scorer_path = 'src/eval/scorer.pl'

rucoref_train = rueval.RuCorefCorpus(multeast, rueval)
rucoref_test = rueval.RuCorefCorpus(multeast, rueval)

with open('../a_dataset_rucoref/Documents.txt', 'r') as f:
    fields = f.readline().strip('\r\n').split('\t')
    setDoc = set()
    for row in f:
        n = row.strip('\r\n').split('\t')
        setDoc.add(n[0])

doc_test = random.choices(list(setDoc), k = 50)
doc_train = list(setDoc - set(doc_test))

exp_utils.load_corpus(rucoref_train, '../a_dataset_rucoref/Tokens.txt', '../a_dataset_rucoref/Groups.txt', numdoc=doc_train)
print('===========')
exp_utils.load_corpus(rucoref_test, '../a_dataset_rucoref/Tokens.txt', '../a_dataset_rucoref/Groups.txt', numdoc=doc_test)

good_pronouns = {u'я', u'мы',
                 u'ты', u'вы',
                 u'он', u'она', u'оно', u'они',
                 u'мой', 'наш',
                 u'твой', u'ваш',
                 u'его', u'ее', u'их',
                 u'себя', u'свой',
                 u'который'
                }
group_ok = lambda g: g.tag.startswith('N') or (g.tag.startswith('P') and g.lemma[0] in good_pronouns)


class BaselineHeadMatchProClassifier(mentionpair.MentionPairClassifier):
    def __init__(self, scorer_path):
        super(BaselineHeadMatchProClassifier, self).__init__(scorer_path)
        self.groups_match = lambda pair: pair[0].lemma[pair[0].head] == pair[1].lemma[pair[1].head]

    def pair_coreferent(self, pair, groups, words):
        tagset = rucoref_test.tagset

        is_pronoun = lambda w: tagset.pos_filters['pronoun'](w)
        is_deictic_pronoun = lambda w: tagset.extract_feature('person', w) in ('1', '2')

        number_agrees = lambda p: same_grammemmes('number', p, tagset)
        gender_agrees = lambda p: same_grammemmes('gender', p, tagset)

        if is_pronoun(pair[1]):
            heads = [np.words[np.head] if np.type != 'word' else np for np in pair]
            heads_indices = [words.index(head) for head in heads]

            nouns_agr_between = [word for word in words[heads_indices[0] + 1:heads_indices[1]]
                                 if tagset.pos_filters['noun'](word)
                                 and number_agrees((word, pair[1]))
                                 and gender_agrees((word, pair[1]))
                                 ]

        return (
                (is_deictic_pronoun(pair[0]) and self.groups_match(pair))
                or
                (not is_pronoun(pair[0]) and pair[0].lemma[pair[0].head] == pair[1].lemma[pair[1].head])
                or
                (
                        not is_pronoun(pair[0]) and is_pronoun(pair[1])
                        and number_agrees(pair)
                        and gender_agrees(pair)
                        and len(nouns_agr_between) == 0
                )
        )




gs_mentions, gs_group_ids = coref_utils.get_gs_groups(rucoref_test)
gs_groups = gs_mentions

pred_mentions, pred_group_ids = coref_utils.get_pred_groups(rucoref_test, group_ok)
pred_groups = rucoref_test.groups

pred_mentions_gold_bound, pred_gold_bounds_ids = coref_utils.get_pred_groups_gold_boundaries(rucoref_test, group_ok)
pred_groups_gold_bound = rucoref_test.groups

scores, groups, chains_base = BaselineHeadMatchProClassifier(scorer_path).score(rucoref_test,
                                                                                pred_mentions_gold_bound,
                                                                                pred_groups_gold_bound,
                                                                                metrics=('muc',), heads_only=False)


coref_utils.get_score_table(BaselineHeadMatchProClassifier(scorer_path), rucoref_test, gs_mentions, gs_groups, False)
coref_utils.get_score_table(BaselineHeadMatchProClassifier(scorer_path), rucoref_test, pred_mentions_gold_bound, pred_groups_gold_bound, False)
coref_utils.get_score_table(BaselineHeadMatchProClassifier(scorer_path), rucoref_test, pred_mentions, pred_groups, False)
#coref_utils.print_chains_in_text(rucoref_test, 1, chains_base, pred_mentions_gold_bound)