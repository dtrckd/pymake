
def basic_plot():

    columns = 1
    targets = ['text/nips12/debug/inference-ilda_10_auto_100',
               'text/nips12/debug/inference-lda_cgs_1_auto_100',
               'text/nips12/debug/inference-lda_cgs_2_auto_100',
               'text/nips12/debug/inference-lda_cgs_5_auto_100',
               'text/nips12/debug/inference-lda_cgs_10_auto_100000000', ]
    plot_csv(targets, columns, separate=False)

def complex_plot(spec):
	sep = 'corpus'
	separate = 'N'
	targets = make_path(spec, sep=sep)
	json_extract(targets)
	if sep:
		for t in targets:
			plot_csv(t, spec['columns'], separate=separate, twin=False, iter_max=spec['iter_max'])
	else:
		plot_csv(targets, spec['columns'], separate=True, iter_max=spec['iter_max'])

def make_path(spec, sep=None, ):
    targets = []
    if sep:
        tt = []
    for base in spec['base']:
        for hook in spec['hook_dir']:
            for c in spec['corpus']:
                p = os.path.join(base, c, hook)
                for n in spec['Ns']:
                    for m in spec['models']:
                        for k in spec['Ks']:
                            for h in spec['hyper']:
                                for hm in spec['homo']:
                                    t = 'inference-%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
                                    t = os.path.join(p, t)
                                    filen = os.path.join(os.path.dirname(__file__), "../data/", t)
                                    if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
                                        continue
                                    if sum(1 for line in open(filen)) <= 1:
                                        # empy file
                                        continue
                                    targets.append(t)

                if sep == 'corpus' and targets:
                    tt.append(targets)
                    targets = []

    if sep:
        return tt
    else:
        return targets

# Return dictionary of property for an expe file. (format inference-model_K_hyper_N)
def get_expe_file_prop(target):
    _id = target.split('_')
    model = ''
    st = 0
    for s in _id:
        try:
            int(s)
            break
        except:
            st += 1
            model += s

    _id = _id[st:]
    prop = dict(
        corpus = target.split('/')[-3],
        model = model.split('-')[-1],
        K     = _id[0],
        hyper = _id[1],
        homo = _id[2],
        N     = _id[3],)
    return prop

# Return size of proportie in a list if expe files
def get_expe_file_set_prop(targets):
    c = []
    for t in targets:
        c.append(get_expe_file_prop(t))

    sets = {}
    for p in ('N', 'K'):
        sets[p] = len(set([ _p[p] for _p in c ]))

    return sets

def json_extract(targets):
    l = []
    for t in targets:
        for _f in t:
            f = os.path.join(os.path.dirname(__file__), "../data/", _f) + '.json'
            d = os.path.dirname(f)
            corpus_type = ('/').join(d.split('/')[-2:])
            f = os.path.basename(f)[len('inference-'):]
            fn = os.path.join(d, f)
            try:
                d = json.load(open(fn,'r'))
                l.append(d)
                density = d['density'] # excepte try density_all
                mask_density = d['mask_density']
                #print density
                #print mask_density
                precision = d['Precision']
                rappel = d['Recall']
                K = len(d['Local_Attachment'])
                h_s = d.get('homo_ind1_source', np.inf)
                h_l = d.get('homo_ind1_learn', np.inf)
                nmi = d.get('NMI', np.inf)
                print '%s %s; \t K=%s,  global precision: %.3f, local precision: %.3f, rappel: %.3f, homsim s/l: %.3f / %.3f, NMI: %.3f' % (corpus_type, f, K, d.get('g_precision'), precision, rappel, h_s, h_l, nmi )
            except Exception, e:
                print e
                pass

    print
    if len(l) == 1:
        return l[0]
    else:
        return l

if __name__ ==  '__main__':
    block = True
    conf = argParse()

    spec = dict(
        base = ['networks'],
        hook_dir = ['debug5/'],
        #corpus   = ['kos', 'nips12', 'nips', 'reuter50', '20ngroups'],
        #corpus   = ['generator/Graph1', 'generator/Graph2', 'clique3'],
        #corpus   = ['generator/Graph3', 'generator/Graph4'],
        corpus   = ['generator/Graph4', 'generator/Graph10', 'generator/Graph12', 'generator/Graph13'],
        columns  = ['perplexity'],
        #models   = ['ibp', 'ibp_cgs'],
        #models   = ['ibp_cgs', 'immsb'],
        ##models   = ['immsb', 'mmsb_cgs'],
        models   = [ 'ibp', 'immsb'],
        #Ns       = [250, 1000, 'all'],
        Ns       = ['all',],
        #Ks       = [5, 10, 15, 20, 25, 30],
        Ks       = [5, 10],
        #Ks       = [5, 10, 30],
        #Ks       = [10],
        #homo     = [0,1,2],
        homo     = [0],
        hyper    = ['fix', 'auto'],
        #hyper    = ['auto'],
        iter_max = 500 ,
    )

    sep = 'corpus'
    separate = 'N'
    #separate = 'N' and False
    targets = make_path(spec, sep=sep)
    json_extract(targets)
    exit()
    if sep:
        for t in targets:
            plot_csv(t, spec['columns'], separate=separate, twin=False, iter_max=spec['iter_max'])
    else:
        plot_csv(targets, spec['columns'], separate=True, iter_max=spec['iter_max'])


    ### Basic Plots
    #basic_plot()

    display(block)
