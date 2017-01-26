def plot_K_dyn(sep=' ', columns=[0,'K', 'K_hist'], target_dir='K_dyn'):
    bdir = os.path.join(os.path.dirname(__file__), "../../output", target_dir)

    # figure
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    fig = plt.figure()
    fig.canvas.set_window_title(target_dir)

    # for compared curves
    extra_c = []
    for i, column in enumerate(columns):

        # subplot
        ax1 = fig.add_subplot(2, 2, i+1)
        if column is 'K':
            plt.title('LL end point')
            ax1.set_xlabel('run')
            ax1.plot(extra_c, marker=next(markers))
            continue
        elif column is 'K_hist':
            plt.title('K distribution')
            ax1.set_xlabel('K')
            ax1.set_ylabel('P(K)')
            bins = int( len(set(k_order)) * 1)
            #k_order, _ = np.histogram(k_order, bins=bins, density=True)
            ax1.hist(k_order, bins, normed=True, range=(min(k_order), max(k_order)))
            continue
        else:
            ylabel, label = tag_from_csv(i)
            plt.title('Likelihood consistency')
            ax1.set_xlabel('iterations')
            ax1.set_ylabel(ylabel)

        k_order = []
        # Assume one mcmc file by directory
        for dirname, dirnames, filenames in os.walk(bdir):
            if not 'mcmc' in filenames:
                continue

            filen = os.path.join(dirname, 'mcmc')
            with open(filen) as f:
                data = f.read()

            data = filter(None, data.split('\n'))
            data = [x.strip() for x in data if not x.startswith(('#', '%'))]
            _k = data[csv_row('K')][-1]
            k_order.append(int(_k))
            curve = [row.split(sep)[column] for row in data]
            curve = np.ma.masked_invalid(np.array(curve, dtype='float'))
            extra_c.append(curve.max())
            ax1.plot(curve, marker=next(markers), label=_k)
            #leg = ax1.legend()

    plt.draw()
