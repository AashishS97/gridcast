# Trial: serving-2026-10-27 (challenger) vs the champion

Written 2026-11-27 17:32 UTC. Paired comparison on origins from 2026-10-27
(the challenger's cutoff), both models scored against the same final actuals.
Mean pinball over p10/p50/p90; decide after 14 pairs, end at 28; non-inferiority margin +2%.

**Decision: wait** -- 2 paired origin(s), need 14

- paired origins: 2

Promotion = merging the pull request that sets `train_cutoff` in
`configs/serving.toml`; CI then deploys this bundle. Rollback = revert that
commit: every bundle stays an immutable release.
