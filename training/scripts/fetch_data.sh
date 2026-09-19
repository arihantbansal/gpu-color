#!/usr/bin/env bash
# Downloads the xkcd colour survey into training/data/ and loads it into SQLite.
# Sources: https://blog.xkcd.com/2010/05/03/color-survey-results/ (CC0)
set -euo pipefail
cd "$(dirname "$0")/../data"
[ -f rgb.txt ] || curl -sSL -o rgb.txt https://xkcd.com/color/rgb.txt
[ -f colorsurvey.tar.gz ] || curl -sSL -o colorsurvey.tar.gz https://xkcd.com/color/colorsurvey.tar.gz
[ -f mainsurvey_sqldump.txt ] || tar xzf colorsurvey.tar.gz
[ -f mainsurvey.sqlite ] || sqlite3 mainsurvey.sqlite < mainsurvey_sqldump.txt
echo "ready: $(pwd)/mainsurvey.sqlite"
