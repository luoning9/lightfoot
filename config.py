import logging
import os
import sys

from configparser import ConfigParser

CONFIG = True


def initialize_config() -> ConfigParser:
    global CONFIG
    if CONFIG is True:
        config = ConfigParser()
        config.read("./private_config.ini")
        # set environment vars
        if os.getenv('all_proxy') is None:
            os.environ['all_proxy'] = config.get('default','SOCKS_PROXY')
        if os.getenv('OPENAI_API_KEY') is None:
            os.environ['OPENAI_API_KEY'] = config.get('default', 'OPENAI_API_KEY')

        # set global vars
        CONFIG = config

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    return CONFIG



