git submodule init
git submodule update

cd lib/pysvd3
sh make.sh
cd ../..

python setup.py build_ext --inplace
