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
        self.ua = 'Linux / Firefox 44: Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:44.0) Gecko/20100101 Firefox/44.0'

    def getTargetFile(self, repo):
        return  repo['filename'] + '/' + repo['url'].split('/')[-1]

    def getIdFile(self, repo):
        return  repo['filename'] + '/' + repo['filename'] + '.' + repo['ext']

    def wget(self, url, out):
        call([ 'mkdir', '-p', path.dirname(out)])
        call(['wget', url, '-O', out, '--user-agent', self.ua])
        return

    def fetch(self, repo=None):
        if repo is None:
            REPO = self.REPO
        else:
            REPO = [self.REPO[i] for i, isin in enumerate((d['filename'] in repo for d in self.REPO)) if isin is True]
            if len(REPO) == 0:
                print('available repo: %s' % ([r['filename'] for r in self.REPO]))
                exit()

        for repo in REPO:
            # exceture postprocessing // mkdir etc
            self.wget(repo['url'], self.getTargetFile(repo))
            self.postProcess(repo)

        return

    def postProcess(self, repo):
        ''' Post processing :
            * decompress
            * rename
        '''
        target_file = self.getTargetFile(repo)
        id_file = self.getIdFile(repo)

        # Decompress
        if target_file.endswith('.tar.gz'):
            call(['tar', 'zxvf', target_file, '-C',  repo['filename'], '--strip-components', '1'])
            # by chance only manufacturing for decmopress well in filenames...
        elif target_file.endswith('.gz'):
            call(['gzip', '-d', target_file])
            call(['mv', target_file[:-len('.gz')], id_file])
        else:
            call(['mv', target_file, id_file])
        return

if __name__ == '__main__':
    import sys
    repo = sys.argv[1:] if len(sys.argv) > 1 else None
    print(repo)

    df = DataFetcher()
    df.fetch(repo)

