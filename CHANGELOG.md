# Changelog

## v1.1.2 — 2026-08-20

- Consolidated the source archive and all 60 inference-only checkpoint assets
  into one current GitHub release.
- Replaced version-specific public links with the current release URL and the
  version-independent Zenodo concept DOI.
- Retained the checkpoint artifact version (`v1.1.0`) and its hashes unchanged;
  this release does not retrain or repack any model.

- Replaced legacy/custom `.zenodo.json` keys with Zenodo's documented GitHub
  integration schema (`upload_type`, `access_right`, normalized creator name,
  and SPDX-compatible license identifier).
- No formal result or frozen experimental protocol changed.
- Clarified field-level reproduction provenance: 193 numerical fields are
  recomputed from public rows and 39 are verified from frozen summaries.
- Added a public v2 protocol disclosure, family-reference evidence boundary,
  and deterministic verification for all 60 companion checkpoints.
- Hardened checkpoint metadata validation and corrected the reusable
  two-sided bootstrap p-value helper to use a plus-one finite-sample rule.
- Removed an obsolete RC1 protocol copy, a stale file-by-file guide, an
  internal Chinese table-writing note, and an unused mechanism intermediate
  table from the current branch; tagged history retains prior versions.

These changes document and verify the already frozen experiments; no model was
retrained and no new experimental result was created. Earlier online release
objects were retired after this consolidated release was verified.
