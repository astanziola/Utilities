#!/bin/bash

# Adapted from:
# https://unix.stackexchange.com/a/436713

MAX_PROCESSES=16  # Max. number of parallel processes

subprocess() {
	eval subpath="$1"
	echo "${subpath}"
}

for d in */ ; do
	subprocess "\${d}" &

    if [[ $(jobs -r -p | wc -l) -ge $MAX_PROCESSES ]]; then
        # now there are $N jobs already running, so wait here for any job to be finished
        wait -n
    fi
done

# no more jobs to be started but wait for pending jobs
wait


