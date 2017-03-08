#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from pymake.scripts import fit
from pymake import GramExp

GramExp.generate().pymake(fit.Fit)


