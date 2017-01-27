#!/bin/bash

# Principal repository
# * KONECT -- 261 networks of all kind     : http://konect.uni-koblenz.de/networks/
# * SNAP -- leskovec networks, quite big   : https//snap.stanford.edu/data
# * Lovro Subelj                           : http://wwwlovre.appspot.com/support.jsp
# * UCINET IV -- Small network < 100 nodes : http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm

#
# Scrapy -- DB fetch
#

# manufacturing (167, D): http://konect.uni-koblenz.de/networks/radoslaw_email
wget https://www.ii.pwr.edu.pl/~michalski/datasets/manufacturing.tar.gz
# format csv

# fb_uc (1899, D): http://konect.uni-koblenz.de/networks/opsahl-ucsocial
wget http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt
# format tnet

# emaileu (986, D) : https://snap.stanford.edu/data/email-Eu-core.html
wget https://snap.stanford.edu/data/email-Eu-core.txt.gz

#  propro (1870, U): http://konect.uni-koblenz.de/networks/moreno_propro
wget http://moreno.ss.uci.edu/pro-pro.dat

# blogs (1224, D) -- hyperlinks : http://konect.uni-koblenz.de/networks/moreno_blogs
wget http://moreno.ss.uci.edu/blogs.dat

# euroroad  (1174, U) : http://konect.uni-koblenz.de/networks/subelj_euroroad
wget http://wwwlovre.appspot.com/resources/research/networks/bpa/euroroad.net
#. net -> .dat (add DATA: )

