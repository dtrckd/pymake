# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import re, os, json, logging
from collections import OrderedDict
import numpy as np

lgg = logging.getLogger('root')

''' This is Obsolote and will be merge into Gramexp module '''

LOCAL_BDIR = '../../data/' # Last slash(/) necessary.
if not os.path.exists(os.path.dirname(__file__)+'/'+LOCAL_BDIR+'networks/generator/Graph7/debug111111'):
    LOCAL_BDIR = '/media/dtrckd/TOSHIBA EXT/pymake/data/'
    if not os.path.exists(LOCAL_BDIR):
        LOCAL_BDIR = '/home/ama/adulac/workInProgress/networkofgraphs/process/pymake/data/'
        #print ('Error Data path: %s' % LOCAL_BDIR)
        #exit()

_STIRLING_PATH = LOCAL_BDIR + '/../pymake/util/'

### directory/file tree reference
# Default and New values
# @filename to rebuild file
_MASTERKEYS = OrderedDict((
    ('data_type'   , None),
    ('corpus'      , None),
    ('repeat'      , None),
    ('model'       , None),
    ('K'           , None),
    ('hyper'       , None),
    ('homo'        , None),
    ('N'           , 'all'),
))


# Debug put K_end inside the json
#_Key_measures = [ 'g_precision', 'Precision', 'Recall', 'K',
#                 'homo_dot_o', 'homo_dot_e', 'homo_model_o', 'homo_model_e',
#                ]
_Key_measures = [ 'g_precision', 'Precision', 'Recall', 'K',
                 'homo_dot_o', 'homo_dot_e', 'homo_model_o', 'homo_model_e',
                 'f1'
                ]

_New_Dims = [{'measure':len(_Key_measures)}]


def is_empty_file(filen):
    if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
        return True

    with open(filen, 'rb') as f: first_line = f.readline()
    if first_line[0] in ('#', '%') and sum(1 for line in open(filen)) <= 1:
        # empy file
        return True
    else:
       return False

def ext_status(filen, _type):
    nf = None
    if _type == 'pk':
        nf = filen + '.pk'
    elif _type == 'json':
        nf = filen + '.json'
    elif _type in ('inf', 'inference'):
        nf =  filen + '.inf'
    elif _type == 'all':
        nf = dict(pk=filen + '.pk',
                 json=filen + '.json',
                 inference=filen + '.inf')
    return nf


# Obsolete !
def tree_hook(key, value):
    hook = False
    if key == 'corpus':
        if value in ('generator', ):
            hook = True
    return hook


# Obsolete !
def get_conf_from_file(target, mp):
    """ Return dictionary of property for an expe file.
        @mp: map parameters
        format model_K_hyper_N
        @template_file order important to align the dictionnary.
        """
    masterkeys = _MASTERKEYS.copy()
    template_file = masterkeys.keys()
    ##template_file = 'networks/generator/Graph13/debug11/immsb_10_auto_0_all.*'

    # Relative path ignore
    if target.startswith(LOCAL_BDIR):
        target.replace(LOCAL_BDIR, '')

    path = target.lstrip('/').split('/')

    _prop = os.path.splitext(path.pop())[0]
    _prop = path + _prop.split('_')

    prop = {}
    cpt_hook_master = 0
    cpt_hook_user = 0
    # @Debug/Improve the nasty Hook here
    def update_pt(cur, master, user):
        return cur - master + user

    #prop = {k: _prop[i] for i, k in enumerate(template_file) if k in mp}
    for i, k in enumerate(template_file):
        if not k in mp:
            cpt_hook_master += 1
            continue
        pt = update_pt(i, cpt_hook_master, cpt_hook_user)
        hook = tree_hook(k, _prop[pt])
        if hook:
            cpt_hook_user += 1
            pt = update_pt(i, cpt_hook_master, cpt_hook_user)
        prop[k] = _prop[pt]

    return prop

def get_conf_dim_from_files(targets, mp):
    """ Return the sizes of proporties in a list for expe files
        @mp: map parameters """
    c = []
    for t in targets:
        c.append(get_conf_from_file(t, mp))

    sets = {}
    keys_name = mp.keys()
    for p in keys_name:
        sets[p] = len(set([ _p[p] for _p in c ]))

    return sets

def get_json(fn):
    try:
        d = json.load(open(fn,'r'))
        return d
    except Exception as e:
        return None

def forest_tensor(target_files, map_parameters):
    """ It has to be ordered the same way than the file properties.
        Fuze directory to find available files then construct the tensor
        according the set space fomed by object found.
        @in target_files has to be orderedDict to align the the tensor access.
    """
    # Expe analyser / Tabulyze It

    # res shape ([expe], [model], [measure]
    # =================================================================================
    # Expe: [debug, corpus] -- from the dirname
    # Model: [name, K, hyper, homo] -- from the expe filename
    # measure:
    #   * 0: global precision,
    #   * 1: local precision,
    #   * 2: recall

    ### Output: rez.shape rez_map_l rez_map
    if not target_files:
        lgg.info('Target Files empty')
        return None

    #dim = get_conf_dim_from_files(target_files, map_parameters) # Rely on Expe...
    dim = dict( (k, len(v)) if isinstance(v, (list, tuple)) else (k, len([v])) for k, v in map_parameters.items() )

    rez_map = map_parameters.keys() # order !
    # Expert knowledge value
    new_dims = _New_Dims
    # Update Mapping
    [dim.update(d) for d in new_dims]
    [rez_map.append(n.keys()[0]) for n in new_dims]

    # Create the shape of the Ananisys/Resulst Tensor
    #rez_map = dict(zip(rez_map_l, range(len(rez_map_l))))
    shape = []
    for n in rez_map:
        shape.append(dim[n])

    # Create the numpy array to store all experience values, whith various setings
    rez = np.zeros(shape) * np.nan

    not_finished = []
    info_file = []
    print(rez.shape)
    for _f in target_files:
        prop = get_conf_from_file(_f, map_parameters)
        pt = np.empty(rez.ndim)

        print(rez.ndim, prop)

        assert(len(pt) - len(new_dims) == len(prop))
        for k, v in prop.items():
            try:
                v = int(v)
            except:
                pass
            try:
                idx = map_parameters[k].index(v)
            except Exception as e:
                lgg.error(prop)
                lgg.error('key:value error --  %s, %s'% (k, v))
                raise ValueError
            pt[rez_map.index(k)] = idx

        f = os.path.join(os.path.dirname(__file__), LOCAL_BDIR, _f)
        d = get_json(f)
        if not d:
            not_finished.append( '%s not finish...\n' % _f)
            continue

        try:
            pt = list(pt.astype(int))
            for i, v in enumerate(_Key_measures):
                pt[-1] = i
                ### HOOK
                # v:  is the measure name
                # json_v: the value of the measure
                if v == 'homo_model_e':
                    try:
                        json_v =  d.get('homo_model_o') - d.get(v)
                    except: pass
                elif v == 'f1':
                    precision = d.get('Precision')
                    try:
                        recall = d.get('Recall')
                        recall*2
                    except:
                        # future remove
                        recall = d.get('Rappel')
                    json_v = 2*precision*recall / (precision+recall)
                else:
                    if v == 'Recall':
                        try:
                            v * 2
                        except:
                            v = 'Rappel'

                    json_v = d.get(v)
                rez[zip(pt)] = json_v

        except IndexError as e:
            lgg.error(e)
            lgg.error('Index Error: Files are probably missing here to complete the results...\n')

        #info_file.append( '%s %s; \t K=%s\n' % (corpus_type, f, K) )

    lgg.debug(''.join(not_finished))
    #lgg.debug(''.join(info_file))
    rez = np.ma.masked_array(rez, np.isnan(rez))
    return rez

def clean_extra_expe(expe, map_parameters):
    for k in expe:
        if k not in map_parameters and k not in [ k for d in _New_Dims for k in d.keys() ] :
            del expe[k]
    return expe

def make_tensor_expe_index(expe, map_parameters):
    ptx = []
    expe = clean_extra_expe(expe, map_parameters)
    for i, o in enumerate(expe.items()):
        k, v = o[0], o[1]
        if v in ( '*',): #wildcar / indexing ...
            ptx.append(slice(None))
        elif k in map_parameters:
            ptx.append(map_parameters[k].index(v))
        elif type(v) is int:
            ptx.append(v)
        elif type(v) is str and ':' in v: #wildcar / indexing ...
            sls = v.split(':')
            sls = [None if not u else int(u) for u in sls]
            ptx.append(slice(*sls))
        else:
            raise ValueError('Unknow data type for tensor forest')

    ptx = tuple(ptx)
    return ptx


