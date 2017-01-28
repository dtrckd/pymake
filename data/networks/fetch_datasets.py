#!/usr/bin/env python3

# Principal Networks repository
# * KONECT -- 261 networks of all kind     : http://konect.uni-koblenz.de/networks/
# * SNAP -- leskovec networks, quite big   : https//snap.stanford.edu/data
# * Lovro Subelj                           : http://wwwlovre.appspot.com/support.jsp
# * UCINET IV -- Small network < 100 nodes : http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm


from subprocess import call
from os import path

class DataFetcher(object):

    """ Meaning of attribute od datasets
        * url : remote path
        * filename : filename on disk
        * ext : impact how the dataset is parsing (Warning : see frontendnetwork)
    """
    #
    # Scrapy -- DB fetch
    # Do
    # * auto detect format / postprocessing
    #

    REPO = [
        dict( # manufacturing (167, D): http://konect.uni-koblenz.de/networks/radoslaw_email
             url = 'https://www.ii.pwr.edu.pl/~michalski/datasets/manufacturing.tar.gz',
             filename = 'manufacturing',
             ext = 'csv'
            ),

        dict( # fb_uc (1899, D): http://konect.uni-koblenz.de/networks/opsahl-ucsocial
             url = 'http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt',
             filename = 'fb_uc',
             ext = 'txt'
            ),

        dict( # emaileu (986, D) : https://snap.stanford.edu/data/email-Eu-core.html
             url = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
             filename = 'emaileu',
             ext = 'txt'
            ),

        dict( # propro (1870, U): http://konect.uni-koblenz.de/networks/moreno_propro
             url = 'http://moreno.ss.uci.edu/pro-pro.dat',
             filename = 'propro',
             ext = 'dat'
            ),

        dict ( # blogs (1224, D) -- hyperlinks : http://konect.uni-koblenz.de/networks/moreno_blogs
              url = 'http://moreno.ss.uci.edu/blogs.dat',
             filename = 'blogs',
             ext = 'dat'
             ),

        dict( # euroroad  (1174, U) : http://konect.uni-koblenz.de/networks/subelj_euroroad
             url = 'http://wwwlovre.appspot.com/resources/research/networks/bpa/euroroad.net',
             filename = 'euroroad',
             ext = 'dat',
             #postpross =  (add DATA: )
            ),
    ]

    def __init__(self):
        pass

    def getPath(self, repo):
        p = repo['filename']
        f = p + '/' + p + '.' + repo['ext']
        return f

    def fetch(self):
        for repo in self.REPO:
            # exceture postprocessing // mkdir etc
            self.wget(repo['url'], self.getPath(repo))
            # exceture preprocessing // format processing / unzip

    def wget(self, url, out):
        call([ 'mkdir', '-p', path.dirname(out)])
        call(['wget', url, '-O', out])

if __name__ == '__main__':
    df = DataFetcher()
    df.fetch()

