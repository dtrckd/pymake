
_gram = [
    '--reset', dict(action='store_true', help='Reset index.'),
    '--limit', dict(type=int, dest='number_results'),
    '--extract-structure', dict(action='store_true', help='Extract structure information.'),
    '--path', dict(nargs='*', help='Reset index'),
]


