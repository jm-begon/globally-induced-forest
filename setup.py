# -*- coding: utf-8 -*-
"""
setup script (largely inspired on scikit-learn's)
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__version__ = 'dev'

import os
import shutil
from distutils.command.clean import clean as Clean

def main_dir():
    return "gif"

class CleanCommand(Clean):
    description = "Remove build directories, and compiled file in the source tree"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk(main_dir()):
            for filename in filenames:
                if (filename.endswith('.so') or filename.endswith('.pyd')
                             or filename.endswith('.dll')
                             or filename.endswith('.pyc')
                             or filename.startswith('.DS_Store')):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    #config.set_options(ignore_setup_xxx_py=True,
    #                   assume_default_configuration=True,
    #                   delegate_options_to_subpackages=True,
    #                   quiet=True)

    config.add_subpackage(main_dir())

    return config


def setup_package():
    long_desc = ""
    try:
        with open('README.md') as f:
            long_desc = f.read()
    except:
        pass
    lic = ""
    try:
        with open('LICENSE') as f:
            lic = f.read()
    except:
        pass
    metadata = dict(name='Gif',
                    author='Jean-Michel Begon',
                    author_email='jm.begon@gmail.com',
                    description='Globally Induced Forest (GIF)',
                    version='dev',
                    long_description=long_desc,
                    license=lic,
                    cmdclass={'clean': CleanCommand})
    metadata['configuration'] = configuration
    setup(**metadata)


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup_package()
