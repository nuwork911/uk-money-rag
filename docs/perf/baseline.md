# Performance baseline: BM25-only service (before dense retrieval)

Recorded 2026-10-05 against the live Cloud Run service, so the Week 2 changes
(ONNX Runtime + embedding model + dense index baked into the image) can be compared
like-for-like.

## Deployed artefact

| Field | Value |
|---|---|
| Git commit | `4da525122d39eba89bf026bdafc11df841639531` |
| Image digest | `sha256:b106c0ef2902d7dc15df15a04962335ab0c2b1868f6191ee1ee28a274d8e538b` |
| Image size (Artifact Registry, **compressed**) | 57,472,005 bytes (57.5 MB) |
| Corpus | 5 GOV.UK guides, 108 chunks (built at deploy time, see note) |
| Region / config | europe-west2, scale-to-zero, max 2 instances |

## Latency (`GET /health`)

| Measure | Value | n |
|---|---|---|
| Server-side, warm | 3.7–5.9 ms | 3 |
| Server-side, cold start | 3.25 s | 1 |

## Method

Server-side numbers are `httpRequest.latency` from Cloud Run request logs:

    gcloud logging read 'resource.type="cloud_run_revision"
      AND resource.labels.service_name="ukmoney-rag" AND httpRequest.requestUrl:"/health"' \
      --limit 6 --freshness 1h --format='table(timestamp,httpRequest.latency,httpRequest.status)'

Client-side `curl` timings were also taken (0.69 / 3.40 / 0.99 s for warm calls) but are
dominated by network distance to London: time-to-first-byte exceeded server latency by
0.4–3.2 s. They are **not** used for comparisons.

## Limitations

- Small samples (n=3 warm, n=1 cold). Good enough to detect a large regression, not to
  claim precise percentiles. A proper load test is out of scope for this project.
- Compare image sizes registry-to-registry; local `docker images` reports uncompressed size.
- The corpus in this image was fetched live during the deploy (`corpus_created_at`
  2026-10-03T22:01:12Z, deployed 22:02:07Z). Fixed in Week 2: images are built from the
  committed `data/snapshot/`.
