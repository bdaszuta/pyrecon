#!/bin/bash
set -euo pipefail

# =============================================================================
# fix for script pathing [with source] [From SE#59895]
export OLD_PWD=${PWD}
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
   # if $SOURCE was a relative symlink, we need to resolve it relative to the
   # path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
export DIR_PACKAGE=${DIR}

cd ${DIR_PACKAGE}
# =============================================================================

# =============================================================================

# export PYTHONPYCACHEPREFIX="/tmp/.pyrecon_cache/"

# test documentation
python3 -m pytest tests/
python3 -m mypy pyrecon/
# python3 -m pylint pyrecon/
python3 -m pytest --doctest-modules pyrecon/
# python3 -m mypy --strict pyrecon/

ruff check pyrecon/
ruff check tests/
ruff check hydro_driver/
ruff check viz/

# =============================================================================

# =============================================================================
cd ${OLD_PWD}
# =============================================================================
