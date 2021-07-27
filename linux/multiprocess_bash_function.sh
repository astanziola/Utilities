#!/bin/bash

MAX_PROCESSES=16

subprocess() {
	eval subpath="$1"
	echo "${subpath}"
}

rootdir="$(pwd)"
for d in */ ; do
	((i=i%MAX_PROCESSES)); ((i++==0)) && wait
	
	fullpath="$rootdir/$d"
	
	subprocess "\${fullpath}" &
done

