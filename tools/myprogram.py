#! /usr/bin/env python

import os
from mypython.my_module import MyClass

os.environ["PROJECT_NAME"] = "My Program"
os.environ["PROJECT_VERSION"] = "1.0.0"

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sharedir = rootdir+"/share"

from mypython.usage import Usage
usage = Usage(motd=sharedir+"/motd")
usage.add_argument('mandatory_parameter', default = 0)
usage.add_argument('--optional-parameter', type=float, default = 0.5)
usage.add_argument('--boolean-parameter', action='store_true')

args = usage.parse_args()

from mypython.logging import Logger
if __name__ == '__main__':

    usage.print_motd()

    logger = Logger("My Program", verbose=args.verbose)
    logger.debug(args)
    logger.message(MyClass())