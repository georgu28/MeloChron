"""Adoption prediction: does a user return to a track they just met?

A separate subpackage from the next-track work in ``melochron/`` because it is
a different task on a different corpus, not a variant of the same one. The unit
of prediction is a single *first encounter* of a (user, track) pair, and the
label is binary: does that pair recur inside an adoption horizon.
"""
