#! /bin/bash

# This file is used by docker-compose.yml as container warmup.

[[ $_ != $0 ]] && REALPATH=`dirname $(readlink -f ${BASH_SOURCE[0]})` || REALPATH=`dirname $(readlink -f $0)`
cd $REALPATH

make modules