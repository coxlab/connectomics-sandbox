gusti1 = [
    #[
        #('fbcorr',
         #{'initialize': {'filter_shape': (5, 5),
                         #'generate': ('random:uniform', {'rseed': 42}),
                         #'n_filters': 48},
          #'kwargs': {'max_out': None, 'min_out': 0}}),
        #('lpool', {'kwargs': {'ker_shape': [2, 2], 'order': 10, 'stride': 2}}),
    #],
    #[
        #('fbcorr',
         #{'initialize': {'filter_shape': (4, 4),
                         #'generate': ('random:uniform', {'rseed': 42}),
                         #'n_filters': 48},
          #'kwargs': {'max_out': None, 'min_out': 0}}),
        #('lpool', {'kwargs': {'ker_shape': [2, 2], 'order': 8, 'stride': 2}}),
    #],
    [
        ('fbcorr',
         {'initialize': {'filter_shape': (4, 4),
                         'generate': ('random:uniform', {'rseed': 42}),
                         'n_filters': 48},
          'kwargs': {'max_out': None, 'min_out': 0}}),
        ('lpool', {'kwargs': {'ker_shape': [2, 2], 'order': 8, 'stride': 2}}),
    ],
    [
        ('fbcorr',
         {'initialize': {'filter_shape': (4, 4),
                         'generate': ('random:uniform', {'rseed': 42}),
                         'n_filters': 48},
          'kwargs': {'max_out': None, 'min_out': 0}}),
        ('lpool', {'kwargs': {'ker_shape': [2, 2], 'order': 8, 'stride': 2}}),
    ],
    [
        ('fbcorr',
         {'initialize': {'filter_shape': (3, 3),
                         'generate': ('random:uniform', {'rseed': 42}),
                         'n_filters': 2048},
          'kwargs': {'max_out': None, 'min_out': 0}}),
    ],
]
