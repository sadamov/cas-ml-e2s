#!/usr/bin/env bash
# Interactive GH200 kernel: JupyterLab inside the cas-ml container on one compute
# node, for running the model cells by hand instead of rendering the whole
# notebook with run_on_gpu.sh. Take the URL it prints, then in VSCode:
#   Select Kernel > Existing Jupyter Server > that URL
#   bash .instructor/containers/start_jupyter.sh
set -euo pipefail
cd "$(dirname "$0")/.."

srun --account=ab016 --partition=debug --nodes=1 --ntasks=1 \
     --gpus-per-task=1 --cpus-per-task=32 --mem=200G --time=01:30:00 \
     --environment="$(pwd)/containers/cas-ml.toml" \
     bash -lc '
set -e
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
# Fresh token each run: the port is reachable from anywhere on the cluster network,
# and a token in a committed script would be no protection at all.
TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
echo "=== CONNECT VSCODE TO: http://$(hostname):8888/?token=$TOKEN ==="
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --IdentityProvider.token="$TOKEN"
'
