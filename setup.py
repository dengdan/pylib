#!/usr/bin/env python

import os
from distutils.core import setup

pkgs = os.listdir('src')
setup(name='pylib',
      version='0.1',
      description='python library for my convinience',
      author='Daniel',
      author_email='dengdan890730@163.com',
      url='git@github.com:dengdan/pylib.git',
      package_dir = {'': 'src'},
      packages=pkgs
     )
