import logging
import os
import sys

from configparser import ConfigParser

CONFIG = True


def initialize_config() -> ConfigParser:
    global CONFIG
    if CONFIG:
        config = ConfigParser()
        config.read("./private_config.ini")
        # set environment vars
        if not(os.getenv('all_proxy')):
            os.environ['all_proxy'] = config.get('default','SOCKS_PROXY')
        if not(os.getenv('OPENAI_API_KEY')):
            os.environ['OPENAI_API_KEY'] = config.get('default', 'OPENAI_API_KEY')

        # set global vars
        CONFIG = config

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    return CONFIG



