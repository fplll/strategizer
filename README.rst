BKZ 2.0 Strategy Search
=======================

Search for BKZ 2.0 lattice-reduction strategies using `fplll <https://github.com/fplll/fplll>`_ and `fplll <https://github.com/fpylll/fpylll>`_.

Getting Started
---------------

We assume you are using a `virtualenv <https://virtualenv.readthedocs.org/>`_ for isolating Python build environments. Then, to install ``fplll`` and ``fpylll`` run
 
   .. code-block:: bash

     $ (strategizer) ./setup.sh

Then run
     
   .. code-block:: bash

     $ (strategizer) export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib"

to allow Python to find ``fplll``.

We need to estimate how many nodes can be enumerated in a second, for this run:

   .. code-block:: bash

     $ (strategizer) python ./set_mdc.py

To run the strategy search, first check the help

   .. code-block:: bash

     $ (strategizer) python ./strategize.py --help

and then run e.g.

   .. code-block:: bash

     $ (strategizer) python ./strategize.py --max-block-size 40 --threads 2 --samples 128

To compare two different strategies, try:

   .. code-block:: bash

     $ (strategizer) python ./compare.py A.json B.json

     
Attribution & License
---------------------

This software is written by:

- Martin R. Albrecht
- Léo Ducas
- Marc Stevens

and licensed under the GPLv2+.
  
The best way to contact the authors is by contacting the `fplll-devel <fplll-devel@googlegroups.com>`_ mailinglist.

  
