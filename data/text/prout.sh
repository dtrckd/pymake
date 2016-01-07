#!/bin/bash
hook="$1"
corpus="$2"

if [ -z $hook ]; then
    hook="debug"
fi
if [ -z $ ]; then
    corpus=*
fi

######
find   $corpus*/$hook* -type f | xargs wc
######

# Rename a part of file path.
#find */debug* -type f | parallel 'f="{}" ; mv -- {} ${f/_auto_/_fix_}'


