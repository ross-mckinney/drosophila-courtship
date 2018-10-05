
# drosophila-courtship

**An app and python package for the tracking, classification, and analysis of fly behaviors.**

## Repo Structure

<pre>
|-courtship
    |-app         # code for generating GUI
    |-ml          # code for creating feature matrices and behavioral classifiers
    |-stats       # code for calculating stats about flies and their behaviors
    |-plots       # code for generating plots

|-docs        # sphinx docs
|-tests       # tests
</pre>

## Installation

1. Clone and navigate to this repo in a terminal
2. Setup an environment using:

~~~bash
conda env create -f environment.yml
~~~

3. Activate environment:

~~~bash
activate courtship  # or 'source activate courtship'
~~~

4. Install drosophila-courtship:

~~~bash
python setup.py install
~~~


## Documentation

*Note: the documentation is currently being updated, but the API should be complete.*

Using a terminal, navigate into the docs folder of this repo, then call `make`:

~~~bash
make html
~~~

This will generate an html file with all of the most recent documentation. This should be located in 'docs/source/build/index.html'.
