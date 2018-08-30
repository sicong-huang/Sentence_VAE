#!/bin/bash

wget http://nlp.stanford.edu/data/glove.6B.zip -P ../embedding/
unzip ../embedding/glove.6B.zip -d ../embedding/glove.6B/
rm ../embedding/glove.6B.zip
