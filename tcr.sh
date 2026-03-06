#!/bin/bash
# TCR: Test && Commit || Revert
MSG="${1:-TCR: working}"
cd "$(dirname "$0")"
make test 2>&1 && {
    git add -A
    git -c user.name="Ryan" -c user.email="ryansoq@gmail.com" commit -m "$MSG"
    echo "✅ TCR: committed"
} || {
    git checkout -- .
    echo "❌ TCR: reverted"
    exit 1
}
