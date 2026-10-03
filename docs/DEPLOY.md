# Deploying to Google Cloud Run

Every push to `main` runs CI, rebuilds the corpus from GOV.UK, builds an image tagged with the
commit SHA, deploys it to Cloud Run (London, `europe-west2`), and smoke-tests the live URL.
GitHub authenticates to Google with **Workload Identity Federation**: no service-account keys
exist anywhere, so there is nothing to leak.

```
push to main -> ci.yml (lint, mypy, tests) -> ingest -> docker build -> Artifact Registry
             -> Cloud Run revision -> curl /health + /search -> traffic
```

## One-time setup (~20 minutes)

Prerequisites: a Google Cloud account with billing enabled, the `gcloud` CLI, and the repo
pushed to GitHub. Run these from your machine.

```bash
export PROJECT_ID="ukmoney-rag-$RANDOM"            # must be globally unique
export REGION="europe-west2"
export REPO="your-github-user/uk-money-rag"        # owner/name, exactly as on GitHub
export BILLING_ACCOUNT="XXXXXX-XXXXXX-XXXXXX"      # gcloud billing accounts list

# 1. Project + APIs
gcloud projects create "$PROJECT_ID"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com

# 2. Image registry
gcloud artifacts repositories create ukmoney --repository-format=docker --location="$REGION"

# 3. Two identities: the app runs with NO permissions; the deployer can only deploy.
gcloud iam service-accounts create ukmoney-runtime --display-name="Cloud Run runtime (no roles)"
gcloud iam service-accounts create ukmoney-deployer --display-name="GitHub Actions deployer"
RUNTIME_SA="ukmoney-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOY_SA="ukmoney-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${DEPLOY_SA}" --role="roles/run.admin"
gcloud artifacts repositories add-iam-policy-binding ukmoney --location="$REGION" \
  --member="serviceAccount:${DEPLOY_SA}" --role="roles/artifactregistry.writer"
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member="serviceAccount:${DEPLOY_SA}" --role="roles/iam.serviceAccountUser"

# 4. Workload Identity Federation, admitting only THIS repository
gcloud iam workload-identity-pools create github --location=global --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc uk-money-rag \
  --location=global --workload-identity-pool=github --display-name="uk-money-rag" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${REPO}'"

POOL_ID=$(gcloud iam workload-identity-pools describe github --location=global --format="value(name)")
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${REPO}"

PROVIDER=$(gcloud iam workload-identity-pools providers describe uk-money-rag \
  --location=global --workload-identity-pool=github --format="value(name)")

# 5. Tell GitHub where to deploy. These are identifiers, not secrets.
gh variable set GCP_PROJECT_ID   --body "$PROJECT_ID"   --repo "$REPO"
gh variable set GCP_WIF_PROVIDER --body "$PROVIDER"     --repo "$REPO"
gh variable set GCP_DEPLOY_SA    --body "$DEPLOY_SA"    --repo "$REPO"
gh variable set GCP_RUNTIME_SA   --body "$RUNTIME_SA"   --repo "$REPO"
```

No `gh` CLI? Add the same four values under the repo's **Settings → Secrets and variables →
Actions → Variables**.

Then push to `main` (or run the `deploy` workflow manually). The live URL appears in the
workflow summary.

## Cost controls (do these before the first deploy)

- **Budget alert**: Console → Billing → Budgets & alerts → create a small monthly budget
  with email alerts. Alerts notify you; they don't stop spending.
- **Scale limits** are already in `deploy.yml`: `--min-instances=0` (scale to zero when idle)
  and `--max-instances=2` (caps runaway traffic).
- **Old images**: each deploy pushes a new image. Add an Artifact Registry cleanup policy
  (Console → Artifact Registry → `ukmoney` → Cleanup policies) to keep only the latest ~10.

## Operating it

| Task | How |
|---|---|
| Check what's live | `curl $URL/health` (git SHA + corpus hash) |
| Logs | Console → Cloud Run → `ukmoney-rag` → Logs, or `gcloud run services logs read ukmoney-rag --region europe-west2` |
| Roll back | `gcloud run services update-traffic ukmoney-rag --region europe-west2 --to-revisions=REVISION=100` |
| Tear down | `gcloud projects delete $PROJECT_ID` |

## Test the image locally first

```bash
make ingest && make docker-build && make docker-run
curl localhost:8080/health
```
