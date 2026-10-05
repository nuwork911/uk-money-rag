#!/usr/bin/env bash
# Offline smoke test: runs the image with NO network and checks /health and
# dense /search from inside the container. Proves the model and index are baked in.
set -euo pipefail

IMAGE="${1:-ukmoney-rag:local}"
NAME="ukmoney-smoke"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network none "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

py() { docker exec "$NAME" python -c "$1"; }

echo "waiting for /health ..."
for i in $(seq 1 60); do
  if py "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)" 2>/dev/null; then
    echo "healthy after ~${i}s"; break
  fi
  if [ "$i" -eq 60 ]; then echo "FAILED: not healthy in 60s"; docker logs "$NAME"; exit 1; fi
  sleep 1
done

echo "== /health"
py "
import json, urllib.request
h = json.load(urllib.request.urlopen('http://127.0.0.1:8080/health'))
print(json.dumps(h, indent=2))
assert 'dense' in json.dumps(h.get('retrievers')), 'dense retriever not enabled'
"

echo "== /search?mode=dense"
py "
import json, urllib.request
r = urllib.request.urlopen('http://127.0.0.1:8080/search?q=state+pension+age&mode=dense')
assert r.status == 200, r.status
print(json.dumps(json.load(r), indent=2)[:1500])
"
echo "== index provenance (embeddings_sha256 depends on CPU; see ADR-0002)"
py "
import json, os, pathlib
root = pathlib.Path(os.environ['UKMONEY_INDEX_ROOT'])
m = json.loads((root / (root / 'LATEST').read_text().strip() / 'manifest.json').read_text())
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), 'unknown')
print('index_id', m['index_id'], '| embeddings_sha256', m['embeddings_sha256'][:12], '| cpu', cpu)
"
echo "SMOKE OK (offline, --network none)"
