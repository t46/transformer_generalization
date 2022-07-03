import json


DATA_PATH = 'cache/CFQ/cfq/dataset.json'
# Only 'cache/CFQ/cfq/dataset.json' contain text information.
# All split dataset contains only indices of texts.
DATA_ID = 0
SAVE_TEXT_PATH = f'results/inspect_raw_data_{DATA_ID}.txt'

json_data = open(DATA_PATH)
raw_data = json.load(json_data)
# len(raw_data) = 239357
# raw_data[0].keys() = dict_keys(['complexityMeasures', 'expectedResponse',
#                                 'expectedResponseWithMids', 'question',
#                                 'questionPatternModEntities', 'questionWithBrackets',
#                                 'questionWithMids', 'ruleIds', 'ruleTree', 'sparql',
#                                 'sparqlPattern', 'sparqlPatternModEntities'])

with open(SAVE_TEXT_PATH, mode='w') as f:
    for key, value in raw_data[DATA_ID].items():
        f.write(str(key) + '\n')
        f.write(str(value) + '\n')
        f.write('\n')

# question = raw_data[0]['question']
# # "Did  Jackie's female actor edit and produce Rad Plaid"
# sparql = raw_data[0]['sparql']
# # 'SELECT count(*) WHERE {\n?x0 ns:film.actor.film/ns:film.performance.character ns:m.011n3bs6
# # .\n?x0 ns:film.editor.film ns:m.0_mhbxp .\n?x0 ns:film.producer.film|ns:film.production_company.films
# # ns:m.0_mhbxp .\n?x0 ns:people.person.gender ns:m.02zsn\n}'
# complexity_measure = raw_data[0]

