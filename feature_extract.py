import os
import json
import re
import numpy as np
import csv
from sklearn.ensemble import GradientBoostingClassifier
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
onlyfiles.remove('gold.json')
onlyfiles.remove('feature_extract.py')
onlyfiles.remove('features.csv')

gold = {}
with open('gold.json') as goldfile:
    content = goldfile.read()
    gold = json.loads(content)

all_topics = {}

def addTopics(filename):
    with open(filename) as f:
        content = f.read()
        analysis = json.loads(content)


        # add to keywords dict
        call_topics = analysis['topics']
        for topic in call_topics['customer']:
            if topic not in all_topics:
                z = len(all_topics)
                all_topics[topic] = z

for file in onlyfiles:
    addTopics(file)

partitions = 5
def extractFile(filename, training_set, csv_lines):
    ## Potential Feature List ##
    # avg customer seg length
    # customer speech percentage relative to agent
    # speaking rate
    # questions?
    # keywords
    # crosstalk time?

    with open(filename) as f:
        content = f.read()
        analysis = json.loads(content)

        def getFeatures(analysis_part):

            part_features = []
            segments = analysis['segments']
            call_topics = analysis['topics']

            customer_speech_time = 0
            customer_num_speeches = 0
            agent_speech_time = 0
            customer_total_words = 0
            customer_words_list = []
            customer_total_sentiment = 0
            customer_num_sentences = 0
            topic_array = [0] * len(all_topics)

            for topic in call_topics['customer']:
                topic_array[all_topics[topic]] = 1

            for seg in segments:
                if seg['sentences'][0]['speaker'] == 'customer':
                    customer_speech_time += (seg['stop'] - seg['start'])
                    customer_num_speeches += 1
                    for sentence in seg['sentences']:
                        if 'mag' in sentence and 'pol' in sentence and sentence['mag']*sentence['pol'] != 0:
                            words = re.sub('<noise>', '', sentence['text']).split(' ')
                            customer_total_words += len(words)
                            customer_words_list.append(words)
                            customer_total_sentiment += sentence['mag']*sentence['pol']
                            customer_num_sentences += 1

                elif seg['sentences'][0]['speaker'] == 'agent':
                    agent_speech_time += (seg['stop'] - seg['start'])

            # get avg customer seg length
            avg_customer_speech_length = float(customer_speech_time) / customer_num_speeches
            part_features.append(avg_customer_speech_length)

            customer_words_list.sort(key=lambda sent: -len(sent))
            avg_top_3_speech_length = (float(len(customer_words_list[0])) + len(customer_words_list[1]) +
                                       len(customer_words_list[2])) / 3
            part_features.append(avg_top_3_speech_length)

            # get customer speech percent realtive to agent
            part_features.append(float(customer_speech_time) / (customer_speech_time + agent_speech_time))

            # get speaking rate estimated as total words/total time spent talking
            rate = float(customer_total_words) / customer_speech_time
            part_features.append(rate)

            # get avg of non-zero sentiments
            part_features.append(float(customer_total_sentiment)/customer_num_sentences)

            # get keyword features
            part_features += topic_array
            return part_features


        # partition analysis.speakers&segments into thirds and add features for each third
        start = 0
        total_duration = analysis['duration']['total']
        features = []
        for i in range(partitions):
            analysis_part = {'segments': [], 'topics': analysis['topics']}
            for j in range(start,len(analysis['segments'])):
                seg = analysis['segments'][j]
                if seg['stop'] > float(total_duration)*(i+1) / partitions:
                    start = j
                    break
                analysis_part['segments'].append(seg)
            part_features = getFeatures(analysis_part)
            features += part_features

        mapper = ['LOW', 'HIGH']
        training_set.append((features, gold[filename[:filename.index('.analysis.json')]]))
        if isinstance(csv_lines, list):
            csv_lines.append([mapper[gold[filename[:filename.index('.analysis.json')]]]] + features)

total = len(onlyfiles)
test_number = 1
test_fraction = round(test_number/total)
num_test = int(test_fraction*total) or 1
total_correct = 0
for i in range(0, total - num_test + 1):
    training_files = onlyfiles[:i] + onlyfiles[i+num_test:]
    training_set = []
    for file in training_files:
        extractFile(file, training_set, False)

    X = np.array([row[0] for row in training_set])
    y = np.array([row[1] for row in training_set])

    gbc = GradientBoostingClassifier()
    gbc.fit(X, y)

    test_files = onlyfiles[i:i+num_test]
    test_set = []
    for file in test_files:
        extractFile(file, test_set, False)

    X_test = np.array([row[0] for row in test_set])
    res = gbc.predict(X_test)
    y_test = np.array([row[1] for row in test_set])
    num_right = 0
    for j in range(len(res)):
        if res[j] == y_test[j]:
            num_right += 1

    if float(num_right) / len(res) == 0:
        print onlyfiles[i]
    else:
        print 'correct'
    total_correct += float(num_right) / len(res)

print total_correct, total

if len(sys.argv) > 1:
    print 'writing csv'
    csv_lines = []
    for train_file in onlyfiles:
        training_set = []
        extractFile(train_file, training_set, csv_lines)


    with open('features.csv', 'w') as csvfile:
        feature_writer = csv.writer(csvfile, delimiter=',')
        for line in csv_lines:
            feature_writer.writerow(line.strip('\n'))

