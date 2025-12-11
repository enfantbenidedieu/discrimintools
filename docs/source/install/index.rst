.. _install:

=======
Install
=======

.. tip::
    This page assumes you are comfortable using a terminal and are familiar with package managers. The only prerequisite for installing discrimintools is Python itself.

Virtual environment
~~~~~~~~~~~~~~~~~~~

Install the 64-bit version of Python 3, for instance from the `official website <https://www.python.org/>`_. Now create a `virtual environment (venv) <https://docs.python.org/3/tutorial/venv.html>`_ and install discrimintools.

.. note::
    The virtual environment is optional but strongly recommended, in order to avoid potential conflicts with other packages.

.. code-block:: console

    PS C:\> python -m venv discrimintools-env # create virtual env
    PS C:\> discrimintools-env\Scripts\activate  # activate
    PS C:\> pip install -U discrimintools  # install discrimintools

Version
~~~~~~~

In order to check your installation, you can use.

.. code:: python

    import discrimintools
    print(discrimintools.__version__)

Using an isolated environment such as pip venv or conda makes it possible to install a specific version of discrimintools with pip and conda and its dependencies independently of any previously installed Python packages.

.. note::
    You should always remember to activate the environment of your choice prior to running any Python command whenever you start a new terminal session.

Dependencies
~~~~~~~~~~~~

discrimintools is compatible with python version which supports both dependencies :

.. include:: dependencies_table.rst