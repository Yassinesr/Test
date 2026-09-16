# Frozen manifests

This directory holds the protocol, not data.

| file | produced by | what it is |
|---|---|---|
| `pranet_protocol.json` | `tools/freeze_manifest.py` | SHA-256 of every file **and** of every decoded image, plus sizes, modes and mask positive fractions, for all 2248 image/mask pairs |
| `pranet_protocol.sha256` | same | the same file hashes in `sha256sum` format, so `sha256sum -c` works without this codebase |
| `collisions.json` | `tools/hash_collisions.py` | the cross-split duplicate audit: flag matrix, nearest-neighbour distance distributions, closest pairs, and exact decoded-pixel duplicates |
| `collisions.pairs.csv` | same | every flagged pair, for eyeballing |

**Commit all of them.** They are blocking items 1 and 2 of §6.2 of the brief,
they are currently absent from the literature, and without them a result is
not auditable by anyone else — including you, six months from now.

They are not generated here because the datasets are not redistributable. Run:

```bash
python tools/freeze_manifest.py --root ./dataset --out manifests/pranet_protocol.json
python tools/hash_collisions.py --root ./dataset \
    --manifest manifests/pranet_protocol.json --out manifests/collisions.json
```

Once committed, every training run records which manifest it used, and
`data.verify: hash` refuses to start if the files on disk have drifted from it.
