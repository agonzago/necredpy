# PO Revision -- necredpy Cross-Reference

This branch is the matching feature branch for the PO revision work
in the `network_credibility` repo. The actual revision changes
(salient-price credibility monitor, SR isoquants, interaction panel,
LaTeX edits) all live there:

  https://github.com/agonzago/network_credibility/tree/claude/incorporate-po-feedback-UY4So

See `PO_revision.md` in that repo for the full plan, run order, and
paste-ready LaTeX edits.

## Why a placeholder here

The new `dynare/experiments/baseline_4sec_salient.mod` exercises the
necredpy credibility-block compiler in a configuration we have not
used before: the credibility input is a model variable defined by an
identity in the model block (`monitor = w_mon_1*pi1 + ...`) rather
than the directly observed `pi_cpi`. The parser pipeline already
supports this -- the BE-1304 credibility block compiler resolves
its monitor variable through the networkx dependency graph -- but
this revision is the first time we will rely on that path in
production estimation.

## What to watch for during the salient-model estimation

If the run in step 4 of `network_credibility/PO_revision.md` fails
or produces a NaN log-likelihood, the most likely failure modes
are on the necredpy side:

1. **Monitor not resolved through dependency graph.** The
   credibility block reads `monitor(-1)`. If the parser does not
   recognise `monitor` as a non-observed model variable to be
   substituted via `(I - Omega)^{-1}` style resolution, the lag
   will be aliased to zero and the credibility stock will not
   evolve. Check `developer_guide.md` section on the networkx
   resolution.

2. **Auxiliary-variable expansion for the `monitor(-1)` lag.** The
   parser auto-generates aux variables for lags > 1, but
   `monitor(-1)` is a lag-1 use and should pass through without
   aux. If an aux is incorrectly generated and not added to the
   shock-permutation tracker, `Q_sub` may go rank-deficient.

3. **Equation-ownership trap.** The new identity
   `monitor = w_mon_1*pi1 + w_mon_2*pi2 + ...` adds one row to
   A/B/C/D. The parser permutes rows at parse time; if the new
   row is owned by `monitor` correctly the system should be
   square. Verify with `scripts/test_solution.py`:

   ```
   cd /home/andres/work/credibility/network_credibility \
     && .venv/bin/python scripts/test_solution.py \
        --experiment baseline_4sec_salient
   ```

If any of those break, the fix is here in `necredpy`, not in the
.mod file.
