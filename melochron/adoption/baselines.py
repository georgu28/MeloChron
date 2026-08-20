"""The baselines the model has to beat, fitted on train and nothing else.

The brief's ordering rule -- baselines before model -- exists because the
previous version of this project shipped a model that only looked good against a
weak comparison. So these are built to be hard rather than to be beaten.

Two of them are not in the brief and are here because Phase 1's measurements put
them there:

* **user-prior.** At a 0.36 base rate with wide per-user variation, "this user
  adopts 60% of what they meet" is most of the signal, and a model that cannot
  beat it has learned nothing about tracks. The brief assumed a rare label,
  where a per-user rate would be far weaker.
* **user x item.** The two priors together are the honest thing to beat; either
  alone is a straw man.

`artist-affinity`, the brief's designated adversary, is not buildable -- this
dataset carries no artist. `item-adoption-rate` replaces it and is arguably
harder, since it is fitted directly on the target rather than on a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Pseudocounts swept when tuning shrinkage. Wide, because the right amount of
#: smoothing differs by an order of magnitude between users (hundreds of
#: encounters each) and items (tens of thousands each).
#:
#: The top of this grid is deliberately far past anything plausible. The first
#: run of Phase 2 chose 300.0 for *both* priors -- the largest value on offer --
#: which means the optimum was never bracketed and the tuning was reporting a
#: boundary, not a minimum. `fit_priors` now says so out loud when that happens.
PSEUDOCOUNT_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0)


@dataclass
class Priors:
    """Shrunk per-user and per-item adoption rates, plus the global floor."""

    global_rate: float
    user_rate: np.ndarray  # [n_users]
    item_rate: np.ndarray  # [n_tracks]
    user_seen: np.ndarray  # [n_users] train encounters behind each rate
    item_seen: np.ndarray  # [n_tracks]
    user_pseudocount: float
    item_pseudocount: float

    def summary(self) -> dict:
        return {
            "global_rate": round(self.global_rate, 4),
            "user_pseudocount": self.user_pseudocount,
            "item_pseudocount": self.item_pseudocount,
            "users_with_history": int((self.user_seen > 0).sum()),
            "users_without_history": int((self.user_seen == 0).sum()),
            "items_with_history": int((self.item_seen > 0).sum()),
            "items_without_history": int((self.item_seen == 0).sum()),
        }


def shrunk_rate(
    keys: np.ndarray, labels: np.ndarray, size: int, prior: float, pseudocount: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-key adoption rate, pulled toward ``prior`` by ``pseudocount``.

    Without shrinkage a user with two train encounters and two adoptions scores
    1.0, which is not evidence of anything. The pseudocount is how many
    prior-weighted observations every key starts with, so keys with no history
    fall back to the prior exactly -- which is what happens to every held-out
    user, by construction.
    """
    seen = np.bincount(keys, minlength=size).astype(np.float64)
    positives = np.bincount(keys, weights=labels.astype(np.float64), minlength=size)
    return (positives + pseudocount * prior) / (seen + pseudocount), seen


def _validation_cut(encounter_ts: np.ndarray, rows: np.ndarray, frac: float) -> np.ndarray:
    """Split train rows in time, so tuning never looks forward.

    A random split would let the pseudocount be chosen using encounters that
    come after the ones it is applied to, which is the same mistake in miniature
    that the temporal split exists to prevent.
    """
    cut = np.quantile(encounter_ts[rows], 1.0 - frac)
    return encounter_ts[rows] >= cut


def fit_priors(
    user_code: np.ndarray,
    track_code: np.ndarray,
    labels: np.ndarray,
    encounter_ts: np.ndarray,
    train_rows: np.ndarray,
    n_users: int,
    n_tracks: int,
    validation_frac: float = 0.15,
    grid: tuple[float, ...] = PSEUDOCOUNT_GRID,
) -> Priors:
    """Fit the global, per-user and per-item rates on train rows only.

    The pseudocounts are chosen on a *temporal* validation slice inside train,
    scored by log loss. Test rows are not touched at any point.
    """
    is_val = _validation_cut(encounter_ts, train_rows, validation_frac)
    fit_rows, val_rows = train_rows[~is_val], train_rows[is_val]

    global_rate = float(labels[fit_rows].mean())

    def best_pseudocount(keys_all: np.ndarray, size: int) -> float:
        keys_fit = keys_all[fit_rows]
        keys_val = keys_all[val_rows]
        truth = labels[val_rows].astype(np.float64)
        best, best_loss = grid[0], np.inf
        for m in grid:
            rate, _ = shrunk_rate(keys_fit, labels[fit_rows], size, global_rate, m)
            p = np.clip(rate[keys_val], 1e-6, 1 - 1e-6)
            loss = -float(np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))
            if loss < best_loss:
                best, best_loss = m, loss
        return best

    user_m = best_pseudocount(user_code, n_users)
    item_m = best_pseudocount(track_code, n_tracks)

    for name, chosen in (("user", user_m), ("item", item_m)):
        if chosen in (grid[0], grid[-1]):
            print(
                f"  warning: {name} pseudocount {chosen:g} sits at the edge of the "
                f"grid, so the optimum is not bracketed and the value is a floor "
                f"or ceiling rather than a minimum"
            )

    # Refit on all of train once the amount of smoothing is settled.
    global_rate = float(labels[train_rows].mean())
    user_rate, user_seen = shrunk_rate(
        user_code[train_rows], labels[train_rows], n_users, global_rate, user_m
    )
    item_rate, item_seen = shrunk_rate(
        track_code[train_rows], labels[train_rows], n_tracks, global_rate, item_m
    )

    return Priors(
        global_rate=global_rate,
        user_rate=user_rate,
        item_rate=item_rate,
        user_seen=user_seen,
        item_seen=item_seen,
        user_pseudocount=user_m,
        item_pseudocount=item_m,
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_user_item(
    priors: Priors,
    user_code: np.ndarray,
    track_code: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    max_rows: int = 2_000_000,
    seed: int = 0,
):
    """Logistic combination of the two priors, fitted on train.

    Neither prior alone is the thing to beat: a model that beats only the user
    prior may have learned nothing but popularity, and one that beats only the
    item prior may have learned nothing but the listener. Their combination is
    the honest floor.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    rows = train_rows
    if rows.shape[0] > max_rows:
        rows = rng.choice(rows, size=max_rows, replace=False)

    features = np.column_stack(
        [_logit(priors.user_rate[user_code[rows]]), _logit(priors.item_rate[track_code[rows]])]
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels[rows].astype(np.int8))
    return model


def score_user_item(model, priors: Priors, users: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    features = np.column_stack([_logit(priors.user_rate[users]), _logit(priors.item_rate[tracks])])
    return model.predict_proba(features)[:, 1]
