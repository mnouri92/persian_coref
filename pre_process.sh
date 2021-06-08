#!/bin/bash
echo "Build custom kernels."
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo "Flags are seted"
# Linux (pip)
g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#echo "Download word vector"
#wget -nc https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz
#gunzip -f cc.fa.300.vec.gz

#wget -nc http://github.com/sobhe/hazm/releases/download/v0.5/resources-0.5.zip
#unzip -o resources-0.5.zip -d resources

echo "Convert to coNLL"
python skeleton2conll.py $1 train
python skeleton2conll.py $2 dev

echo "Preproccesing for train"
python minimize.py
python get_char_vocab.py

mkdir -p logs
python filter_embeddings.py cc.fa.300.vec train.persian.jsonlines dev.persian.jsonlines
cp cc.fa.300.vec.filtered logs
python cache_elmo.py train.persian.jsonlines dev.persian.jsonlines
