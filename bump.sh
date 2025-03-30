#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ezp_branch_name mpl_branch_name"
    exit 1
fi

if ! command -v git &> /dev/null
then
    echo "git not found"
    exit 1
fi

cd "$(dirname "$0")"

git checkout $1 -- external ezp

cd mpl

git checkout $2 -- mpl

cd ..
