from __future__ import print_function
from calc import calc as real_calc
import download_slack_history
import sys
import zerorpc
import gevent
import app
import json
from dialog_act_classification.m_dialog_act_classifier import Slack_BagOfWords
import logging


class MonokulApi(object):
    legacy_token = None
    def calc(self, text):
        """based on the input text, return the int result"""
        try:
            return real_calc(text)
        except Exception as e:
            return 0.0
    def echo(self, text):
        """echo any text"""
        return text
    def send_slack_token(self, token):
        print("hello send_slack_token")
        legacy_token = token
        ret = download_slack_history.find_slack_channels_and_users(token)
        print("found slack channels")
        return ret
    def run_analyses(self, options):
        download_slack_history.download_slack_history(options.token)
        app.run_analyses(options.channels_to_analyze)
        return "greenlet completed!"


def download_and_analyze(token):
    download_slack_history.download_slack_history(token)
    print("download_slack_history done")
    app.run_analyses()
    print("run_analyses done")
    return "Successfully downloaded and analyzed"


def parse_port():
    return 4242


def main():
    addr = 'tcp://127.0.0.1:' + str(parse_port())
    s = zerorpc.Server(MonokulApi())
    s.bind(addr)
    print('start running on {}'.format(addr))
    s.run()


if __name__ == '__main__':
    main()
