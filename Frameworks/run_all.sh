#!/bin/bash

for dir in */ ; do
    echo "Entering directory: $dir"
    cd "$dir"

    if [[ -f "main - extended.py" ]]; then
        echo "Running: main - extended.py"
        python "main - extended.py"
    elif [[ -f "main - extended - linear.py" ]]; then
        echo "Running: main - extended - linear.py"
        python "main - extended - linear.py"
    else
        echo "No suitable main file found in $dir"
    fi

    cd ..
done
