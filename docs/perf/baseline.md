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

# Week 2: dense retrieval service

Recorded 2026-10-05, same method as above (server-side `httpRequest.latency` from
Cloud Run logs; cold start = first request after 22 minutes idle).

## Deployed artefact

| Field | Value |
|---|---|
| Git commit | `2b53f65e04f2f2d6e45a8fd807ffc27fc4eba8a8` |
| Image digest | `sha256:ba10e352cac9e0fd52b679c8dd8a9446689e9173e1d64f6c7bdef8534f575c4d` |
| Image size (Artifact Registry, **compressed**) | 184,458,474 bytes (184.5 MB), 3.2x the BM25 image |
| Corpus | 49 GOV.UK guides, 803 chunks, from the committed snapshot (`210e53ba…`) |
| Dense index | `ab2eb5d42c02`, built inside the image |
| Config | europe-west2, 1 vCPU, **1 GiB** (was 512 MiB), **startup CPU boost on** (was off) |

## Latency

| Measure | BM25 baseline | Dense service | n |
|---|---|---|---|
| `/health`, warm | 3.7–5.9 ms | 3.5–8.2 ms | 3 |
| `/health`, cold start | 3.25 s | 7.15 s | 1 |
| `/search?mode=dense`, warm | n/a | 15.2–17.7 ms | 3 |
| `/search?mode=dense`, first after cold start | n/a | 37.1 ms | 1 |

## Reading these numbers

- Warm latency is unchanged; a dense search, including embedding the query, costs
  about 15–18 ms server-side.
- Cold start rose by about 3.9 s. Likely contributors: a 3.2x larger image, importing
  onnxruntime and numpy, and creating the ONNX session. Not broken down by phase.
- **Not a controlled comparison:** memory and CPU boost changed in the same deploy as
  dense retrieval, so the cold-start delta cannot be attributed to dense alone.
- The first dense query after startup is slower (37 ms), which is consistent with ONNX
  Runtime's first-inference warm-up.
- Accepted for a demo. `min-instances=1` would remove cold starts but bills an
  always-on instance; not worth trial credit here.
