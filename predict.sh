#!/bin/bash


mkdir /logs
mkdir /logs/final

cd ..

cp $1.data-00000-of-00001 Persian_Coref/logs/final/model.max.ckpt.data-00000-of-00001
cp $1.index Persian_Coref/logs/final/model.max.ckpt.index

cp $2 Persian_Coref/logs

cd Persian_Coref
python demo.py final