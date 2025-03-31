#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "No arguments supplied, using default branches"
    BRANCH_EZP="master"
    BRANCH_MPL="feature-easy-life"
elif [ "$#" -eq 1 ]; then
    echo "One argument supplied, using default mpl branch"
    BRANCH_EZP=$1
    BRANCH_MPL="feature-easy-life"
else
    echo "Two arguments supplied, using custom branches"
    BRANCH_EZP=$1
    BRANCH_MPL=$2
fi

if ! command -v git &> /dev/null
then
    echo "git not found"
    exit 1
fi

cd "$(dirname "$0")"

git fetch origin
git checkout $BRANCH_EZP -- external ezp

cd mpl

git fetch origin
git checkout $BRANCH_MPL -- mpl

cd ..
