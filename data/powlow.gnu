#!/usr/bin/env gnuplot

set datafile separator ';'
set decimalsign locale 'fr_FR.UTF-8'

####################
## storage settings
####################
#set terminal jpeg enhanced font arial 11
#set size 1.0, 0.8
#set output 'powlow.jpg'
#
#set terminal postscript eps color enhanced
###set terminal postscript eps enhanced mono dashed lw 1 "Helvetica" 14 
#set size 1.0, 0.8
#set output 'powlow.eps'
#

corpus='blocksworld'

####################
## plot polow
####################
set title 'stats '.corpus
set xlabel 'nth ngram'
set ylabel 'occurence'
set grid
set autoscale
#set sample 1000
#set logscale x
#set logscale y

# STYLE OF PLOT: points(default), lines, linespoints, steps, boxes, errorbars, impulses, dots etc. (try gnuplot> test)
set style data linespoints
set style line 1 lw 1
set key left top
set pointsize 0.1

#plot "../work/barman/ngram/old/barman.2gram.wfreq" t '2-gram', \
#         "../work/barman/ngram/old/barman.3gram.wfreq"  t '3-gram',\
#         "../work/barman/ngram/old/barman.4gram.wfreq"  t '4-gram',\
#         "../work/barman/ngram/old/barman.5gram.wfreq"  t '5-gram',\
#         "../work/barman/ngram/old/barman.6gram.wfreq"  t '6-gram',\
#         "../work/barman/ngram/old/barman.7gram.wfreq"  t '7-gram',\
#         "../work/barman/ngram/old/barman.8gram.wfreq"  t '8-gram',\
#         "../work/barman/ngram/old/barman.9gram.wfreq"  t '9-gram' 
#plot  using 3 t '2-gram' every 2:2 with points


### test
filen='../work/'.corpus.'/data/pb-solved.data'

set term x11 1 title corpus enhanced
plot filen  using 1 t 'plan size'

set term x11 2 title corpus enhanced
plot filen  using 2 t 'nb actions'

set term x11 3 title corpus enhanced
plot filen  using 3 t 'relevant facts'

set term x11 4 title corpus enhanced
plot filen  using 4 t 'plan size'

set term x11 6 title corpus enhanced
plot filen  using 6 t 'time'

set term x11 8 title corpus enhanced
plot filen  using 8 t 'pb memory'

set term x11 9 title corpus enhanced
plot filen  using 9 t 'search memory'

set term x11 11 title corpus enhanced
plot filen  using 11 t 'visited state'

set term x11 12 title corpus enhanced
plot filen  using 12 t 'closed.size ?'

#replot
pause -1
