#!/bin/bash
set -e

if test "$VIRTUAL_ENV" == ""; then
   echo "Please enable a virtual environment"
   echo "http://docs.python-guide.org/en/latest/dev/virtualenvs/"
   exit 1
fi
   
cd fplll
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" --with-max-parallel-enum-dim=120
make clean
make -j4
make install
cd ..

cd fpylll
pip install Cython
pip install Cython
pip install -r requirements.txt
pip install -r suggestions.txt
export PKG_CONFIG_PATH="$VIRTUAL_ENV/lib/pkgconfig:$PKG_CONFIG_PATH"
python setup.py clean
python setup.py build_ext
python setup.py install
cd ..
