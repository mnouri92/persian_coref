#!/bin/bash

#wget -nc http://github.com/sobhe/hazm/releases/download/v0.5/resources-0.5.zip
#unzip -o resources-0.5.zip -d resources

python skeleton2conll.py $1 test
python minimize_test.py
python get_char_vocab_test.py


mkdir -p /logs
mkdir -p /logs/final

#cd ..

cp $2.data-00000-of-00001 Persian_Coref/logs/final/model.max.ckpt.data-00000-of-00001
cp $2.index Persian_Coref/logs/final/model.max.ckpt.index

#cp $3 Persian_Coref/logs
#cd Persian_Coref
python evaluate.py final
