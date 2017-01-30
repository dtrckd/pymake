def plot_K_fix(sep=' ', columns=[0,'K'], target_dir='K_test'):
    bdir = os.path.join(os.path.dirname(__file__), "../../output", target_dir)

    # figure
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    fig = plt.figure()
    fig.canvas.set_window_title(target_dir)

    # for compared curves
    extra_c = []
    for i, column in enumerate(columns):

        # subplot
        ylabel, label = tag_from_csv(i)
        xlabel = 'iterations' if column == 0 else 'K'
        stitle = 'Likelihood convergence' if column == 0 else 'Likelihood comparaison'
        ax1 = fig.add_subplot(1, 2, i+1)
        plt.title(stitle)
        #ax1.set_title(stitle)
        ax1.set_xlabel(xlabel)
        if  column is 'K':
            support = np.arange(min(k_order),max(k_order)+1) # min max of K curve.
            k_order = sorted(range(len(k_order)), key=lambda k: k_order[k])
            extra_c = np.array(extra_c)[k_order]
            ax1.plot(support, extra_c, marker=next(markers))
            continue
        ax1.set_ylabel(ylabel)

        k_order = []
        # Assume one mcmc file by directory
        for dirname, dirnames, filenames in os.walk(bdir):
            if not 'mcmc' in filenames:
                continue

            _k = dirname.split('_')[-1]
            k_order.append(int(_k))
            filen = os.path.join(dirname, 'mcmc')
            with open(filen) as f:
                data = f.read()

            data = filter(None, data.split('\n'))
            data = [x.strip() for x in data if not x.startswith(('#', '%'))]
            curve = [row.split(sep)[column] for row in data]
            curve = np.ma.masked_invalid(np.array(curve, dtype='float'))
            extra_c.append(curve.max())
            ax1.plot(curve, marker=next(markers), label=_k)
            #leg = ax1.legend()

    plt.draw()
