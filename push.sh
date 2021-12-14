#!/bin/bash

for d in a b c d
do
	scp -r ../EdgePipe haonanwa@$d:~/testquant/
done
