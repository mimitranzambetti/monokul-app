import user_accounting.user_join as user_join
import dialog_act_classification.m_dialog_act_classifier as m_dac
from dialog_act_classification.m_dialog_act_classifier import Slack_BagOfWords
import download_slack_history
import argparse
import datetime as dt
import os
import json
import pickle
import pandas as pd
import zipfile
import sys
import logging




def create_messages_dataframe(messages):
    clf = pickle.load(open(os.path.join('backend', 'dialog_act_classification', 'clf.p'), 'rb'))
    slack_bow = pickle.load(open(os.path.join('backend', 'dialog_act_classification', 'slack_bow.p'), 'rb'))
    
    message_texts = map(lambda m: m[u"text"].encode('ascii','ignore'), messages)
    classifications = m_dac.classify_messages(message_texts, clf, slack_bow)
    classified_messages = []
    ind = 0
    for message in messages:
        new_message = {}
        new_message['text'] = message[u"text"].encode('ascii','ignore')
        new_message['user'] = message[u"user"].encode('ascii','ignore')
        new_message['ts'] = message[u"ts"].encode('ascii','ignore')
        new_message['DAC'] = classifications[ind]
        classified_messages.append(new_message)
        ind += 1
    df = pd.DataFrame(data=classified_messages)
    df.ts = df.ts.apply(lambda ts: dt.datetime.fromtimestamp(float(ts)))
    df = df.set_index('ts')
    df = df.sort_index()
    return df


def week_by_week_analysis(df, users, output_path):
    start_dt = df.loc[df.index.min()].name.replace(minute=0, second=0, microsecond=0)
    # turn start_dt into a Monday
    while start_dt.weekday() is not 0:
        start_dt = start_dt - dt.timedelta(days=1)
    last_dt_recorded = df.loc[df.index.max()].name
    end_dt = start_dt + dt.timedelta(days=7)
    while start_dt < last_dt_recorded:
        messages_to_analyze = df.loc[start_dt:end_dt]
        if not os.path.exists(os.path.join(output_path, 'engagement_analysis')):
            os.makedirs(os.path.join(output_path, 'engagement_analysis'))
        if not os.path.exists(os.path.join(output_path, 'gratitude_analysis')):
            os.makedirs(os.path.join(output_path, 'gratitude_analysis'))
        engagement_output_path = os.path.join(output_path, 'engagement_analysis', dt.datetime.strftime(start_dt, "%Y-%m-%d") + ".csv")
        gratitude_output_path = os.path.join(output_path, 'gratitude_analysis', dt.datetime.strftime(start_dt, "%Y-%m-%d") + ".csv")
        analyze_engagement(messages_to_analyze, users, engagement_output_path)
        analyze_gratitude(messages_to_analyze, users, gratitude_output_path)
        start_dt = end_dt
        end_dt = start_dt + dt.timedelta(days=7)


def month_by_month_analysis(df, users, output_path):
    start_dt = df.loc[df.index.min()].name.replace(minute=0, second=0, microsecond=0)
    # turn start_dt into a Monday
    start_dt = start_dt.replace(day=1)
    last_dt_recorded = df.loc[df.index.max()].name
    end_dt = start_dt + dt.timedelta(days=32)
    end_dt = end_dt.replace(day=1)
    end_dt = end_dt - dt.timedelta(days=1)
    while start_dt < last_dt_recorded:
        messages_to_analyze = df.loc[start_dt:end_dt]
        if not os.path.exists(os.path.join(output_path, 'engagement_analysis')):
            os.makedirs(os.path.join(output_path, 'engagement_analysis'))
        if not os.path.exists(os.path.join(output_path, 'gratitude_analysis')):
            os.makedirs(os.path.join(output_path, 'gratitude_analysis'))
        engagement_output_path = os.path.join(output_path, 'engagement_analysis', dt.datetime.strftime(start_dt, "%Y-%m-%d") + ".csv")
        gratitude_output_path = os.path.join(output_path, 'gratitude_analysis', dt.datetime.strftime(start_dt, "%Y-%m-%d") + ".csv")
        analyze_engagement(messages_to_analyze, users, engagement_output_path)
        analyze_gratitude(messages_to_analyze, users, gratitude_output_path)
        start_dt = end_dt + dt.timedelta(days=1)
        end_dt = start_dt + dt.timedelta(days=32)
        end_dt = end_dt.replace(day=1)
        end_dt = end_dt - dt.timedelta(days=1)


def analyze_engagement(df, users, output_path):
    user_scores = {}
    for user in users:
        user_scores[user] = {}
    engaged_response_tags = ["qy", "sv", "qw", "qo", "ny"]
    engaged_responses = df[df['DAC'].isin(engaged_response_tags)]
    out_file = open(output_path, 'w+')
    for user in user_scores:
        user_engaged_responses = engaged_responses[engaged_responses['user'] == user]
        total_length_of_engagements = len(" ".join(user_engaged_responses.text))
        # for now, their score is just the length of engagements they've had.
        user_scores[user] = total_length_of_engagements
        out_file.write(user.encode('ascii','ignore') + "," + str(user_scores[user]) + "\n")
    out_file.close()


def analyze_gratitude(df, users, output_path):
    user_scores = {}
    for user in users:
        user_scores[user] = {}
    response_tags = ["ft"]
    responses = df[df['DAC'].isin(response_tags)]
    out_file = open(output_path, 'w+')
    for user in user_scores:
        user_responses = responses[responses['user'] == user]
        total_length_of_responses = len(" ".join(user_responses.text))
        # for now, their score is just the length of engagements they've had.
        user_scores[user] = total_length_of_responses
        out_file.write(user.encode('ascii','ignore') + "," + str(user_scores[user]) + "\n")
    out_file.close()


def dac_analyses(rootdir, channels_to_analyze):
    print("Running analyses...")
    channels = []
    total_messages_analyzed = 0

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file not in channels_to_analyze:
                continue
            print(file)
            channel_name = file[:-5]
            channels.append(file)
            all_messages = []
            data = json.load(open(os.path.join(rootdir, file)))
            for obj in data[u"messages"]:
                if obj[u"type"] == u"message" and u"subtype" not in obj:
                    all_messages.append(obj)
                    total_messages_analyzed += 1
            if len(all_messages) < 1:
                continue
            df = create_messages_dataframe(all_messages)
            users = [] # list of all users in the channel
            for message in all_messages:
                if message[u"user"] not in users:
                    users.append(message[u"user"])
            week_by_week_analysis(df, users, os.path.join(os.pardir, 'export', 'by_channel', channel_name, \
                                'weekly'))
            month_by_month_analysis(df, users, os.path.join(os.pardir, 'export', 'by_channel', channel_name, \
                                'monthly'))
    print("done with DAC analysis")
    return total_messages_analyzed


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def run_analyses(channels_to_analyze):
    path_to_analyze = os.path.join(os.pardir, 'intermediate_files', 'slack_messages', 'channels')
    print("calling dac_analysis on: ")
    print(path_to_analyze)
    dac_analyses(path_to_analyze, channels_to_analyze)
    path_to_analyze = os.path.join(os.pardir, 'intermediate_files', 'slack_messages', 'private_channels')
    print("calling dac_analysis on: ")
    print(path_to_analyze)
    dac_analyses(path_to_analyze, channels_to_analyze)
    print("zipping file")
    zipf = zipfile.ZipFile('monokul_export.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(os.path.join(os.pardir, 'export', 'by_channel'), zipf)
    zipf.close()
    print("done zipping")
    print("returning")
    return 0
    


if __name__ == "__main__":
    # user_join.save_user_join('user_accounting/general.json', 'export/user_join.csv')
    # print "start time: "
    # print dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M:%S")
    run_analyses()
    # print "end time:"
    # print dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M:%S")


