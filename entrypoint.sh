#! /bin/bash

# This file is used by docker-compose.yml as container warmup.

[[ $_ != $0 ]] && REALPATH=`dirname $(readlink -f ${BASH_SOURCE[0]})` || REALPATH=`dirname $(readlink -f $0)`
cd $REALPATH

# Build conda environment
ln -snf /mnt/local $HOME/local
ln -snf /mnt/local/run.me $HOME/bootstrap.sh

MOTD=0 make env

source .bashrc
MOTD=0 make vendor

# [..] add more warmup commands here

echo "Container warmup complete. (read-only project directory: $PWD)"
exec "$@"