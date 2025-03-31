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

cd mpl

git fetch origin
git checkout origin/$BRANCH_MPL -- mpl

if ! git diff --quiet; then
  echo "Changes detected in mpl, updating submodules..."
  git add --all && git commit -m "Automatic sync with $BRANCH_MPL" || exit 1
fi

cd ..

git fetch origin
git checkout origin/$BRANCH_EZP -- external ezp CMakeLists.txt

if ! git diff --quiet; then
  echo "Changes detected in ezp, updating submodules..."
  git add --all && git commit -m "Automatic sync with $BRANCH_EZP" || exit 1
fi
