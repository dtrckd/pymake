#!/usr/bin/env gnuplot
#
# plot perfomance comparaison on learning planning

set datafile separator ';'
set decimalsign locale "fr_FR.UTF-8"
corpus='barman'
style_gram='phrase'
fdo='temp.data'
#fd=style_gram.'.10000.4g5.data'
fd="temp0.data"
#corpus='visitall'
#fdo='temp.data'
#fd=style_gram.'.10000.4g5.data'

####################
## storage settings
####################
set term x11 enhanced

#set terminal jpeg enhanced font arial 11
#set size 1.0, 0.8
#set output corpus.'_pl-perf.jpg'
#
#set terminal postscript eps color enhanced
###set terminal postscript eps enhanced mono dashed lw 1 "Helvetica" 14 
#set size 1.0, 0.8
#set output corpus.'_pl-perf.eps'
#
#

set multiplot layout 2,1 title 'learning planning comparaison.'.corpus.' domain'
## Style
set grid
set autoscale 

#set sample 1000
# STYLE OF PLOT: points(default), lines, linespoints, steps, boxes, errorbars, impulses, dots etc. (try gnuplot> test)
set style data linespoints
set style line 1 lw 1
set key left top
#set pointsize 0.1
##

####################
## plot time perf
####################
set title 'performance comparaison'
set xlabel 'complexity (# of pb)'
set ylabel 'search time'
set logscale y


plot '../work/'.corpus.'/data/'.fdo u 1:6 t 'before learning reflexe', \
         '../work/'.corpus.'/data/'.fd u 1:6  t 'after learning reflexe'

####################
## plot plan size
####################
set title 'plan size comparaison'
set xlabel 'complexity (# of pb)'
set ylabel 'plan size'
unset logscale 

plot '../work/'.corpus.'/data/'.fdo u 1:($4<0?1/0:$4) t 'before learning reflexe', \
    '../work/'.corpus.'/data/'.fd u 1:($4<0?1/0:$4)  t 'after learning reflexe'

unset multiplot


pause -1
