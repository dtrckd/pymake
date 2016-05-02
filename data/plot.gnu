#!/usr/bin/env gnuplot

#set datafile separator ';'
#set decimalsign locale 'fr_FR.UTF-8'

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

struct = 'text/'
corpus = 'nips/'

####################
## plot polow
####################
set title struct.corpus
set xlabel 'iterations'
set ylabel 'loglikelihood'
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

filename = struct.corpus.'/debug0/inference-ilda_5_fix_10'
plot filename using 2 t 'll'


#replot
pause -1
