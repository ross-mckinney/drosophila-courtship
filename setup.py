DESCRIPTION = (
    'Package for the tracking, classification, and analysis of courtship ' +
    'behavior in D. melanogaster'
    )
VERSION = '0.1.0'
DISTNAME = 'courtship'
AUTHOR = 'Ross McKinney'
AUTHOR_EMAIL = 'ross.m.mckinney@gmail.com'
LICENSE = 'MIT'

# These can also be found in requirements.txt
REQUIRED_PACKAGES = {
    'alabaster': 'alabaster',
    'matplotlib': 'matplotlib',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'PyQt5' : 'PyQt5',
    'sklearn': 'scikit-learn',
    'skimage': 'scikit-image',
    'scipy': 'scipy',
    'sphinx': 'sphinx',
    'sphinxcontrib': 'sphinxcontrib',
    'motmot.FlyMovieFormat': 'motmot.FlyMovieFormat',
    'pycircstat': 'pycircstat',
    'recommonmark': 'recommonmark',
    'nose': 'nose',
    'xlsxwriter': 'xlsxwriter',
    'xlrd': 'xlrd',
    'pyqtgraph': 'pyqtgraph'
}

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

def check_dependencies():
    """Make sure all necessary packages are present."""
    install = []
    for key, value in REQUIRED_PACKAGES.iteritems():
        try:
            module = __import__(key)
        except ImportError:
            install.append(value)

    return install

if __name__ == '__main__':
    install_requires = check_dependencies()

    setup(
        name=DISTNAME,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        install_requires=install_requires,
        packages=[
                'courtship',
                'courtship.ml',
                'courtship.plots',
                'courtship.stats',
                'courtship.app',
                'courtship.app.dialogs',
                'courtship.app.widgets',
                'courtship.app.icons'
            ],
        entry_points={
            'console_scripts': [
                'courtship-app=courtship.app.entry:main',
            ],
        },
        package_data = {
            'courtship.app': [
                'icons/*.png'
            ],
        }
    )
