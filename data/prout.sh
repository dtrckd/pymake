#!/bin/bash

struct="text"
struct="networks"
hook="$1"
command="$2"

if [ -z $hook ]; then
    hook="debug"
fi

#### Search Files !!!
find $struct/$corpus*/$hook* -type f | grep -iEv "(.json|.pk)" |  xargs wc
#find networks/generator/Graph3/debug4* -type f | xargs wc
#find networks/generator/* -type f | grep -iEv "(.json|.pk)" | grep -iE "(debug5|debug6)" | xargs wc

### Rename a part of file path.
#find */debug* -type f | parallel 'f="{}" ; mv -- {} ${f/_auto_/_fix_}'


