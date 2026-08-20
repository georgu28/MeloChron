"""Guards for the Music4All-Onion parser and the compact corpus.

The fixtures here encode failure modes that are real for this corpus rather
than hypothetical: a file whose schema is undocumented, a timestamp whose unit
must be inferred, rows arriving out of order, and a row count that must be
reproduced exactly. Each test names the wrong number it would produce if the
code regressed, because on a 253M-row corpus none of these fail loudly.
"""

from __future__ import annotations

import bz2
from datetime import UTC, datetime

import numpy as np
import pytest

from melochron.adoption.corpus import (
    CompactCorpus,
    build,
    corpus_stats,
    estimate_rows,
)
from melochron.adoption.onion import read_counts_totals, sniff_schema

# 2013-06-01T00:00:00Z, well inside plausible Last.fm history.
BASE_TS = 1370044800


def write_tsv(path, rows, header=None, compress=False):
    """Write a fixture in the shape the real archive uses."""
    lines = []
    if header:
        lines.append("\t".join(header))
    lines.extend("\t".join(str(f) for f in row) for row in rows)
    text = "\n".join(lines) + "\n"
    if compress:
        path.write_bytes(bz2.compress(text.encode("utf-8")))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def simple_rows(n_users=3, n_tracks=4, plays=2):
    """A deterministic corpus where every pair is played exactly ``plays`` times."""
    rows = []
    ts = BASE_TS
    for u in range(n_users):
        for t in range(n_tracks):
            for _ in range(plays):
                rows.append((f"user{u:04d}", f"track{t:04d}", ts))
                ts += 60
    return rows


class TestSniffSchema:
    def test_headerless_tab_separated_seconds(self, tmp_path):
        path = write_tsv(tmp_path / "events.tsv", simple_rows())
        schema = sniff_schema(path)

        assert schema.has_header is False
        assert schema.delimiter == "\t"
        assert (schema.user_idx, schema.track_idx, schema.ts_idx) == (0, 1, 2)
        assert schema.ts_kind == "seconds"

    def test_header_row_is_detected_and_not_parsed_as_data(self, tmp_path):
        path = write_tsv(
            tmp_path / "events.tsv",
            simple_rows(),
            header=["user_id", "track_id", "timestamp"],
        )
        schema = sniff_schema(path)

        assert schema.has_header is True
        assert schema.columns == ["user_id", "track_id", "timestamp"]
        # Eating the header as data would add a phantom user and track.
        assert schema.sample_rows[0][0] == "user0000"

    def test_timestamp_located_by_content_not_position(self, tmp_path):
        """The timestamp is not always last, and position is a prior, not evidence."""
        rows = [(BASE_TS + i * 60, f"user{i % 2}", f"track{i % 3}") for i in range(8)]
        path = write_tsv(tmp_path / "events.tsv", rows)
        schema = sniff_schema(path)

        assert schema.ts_idx == 0
        assert schema.user_idx == 1
        assert schema.track_idx == 2

    def test_datetime_strings_are_resolved_by_header_name(self, tmp_path):
        """The real corpus stores `2013-01-27 21:42:38`, not an integer.

        A content scan cannot tell a datetime string from an opaque id, so with
        the timestamp moved off the last position a positional fallback would
        silently name `track_id` the timestamp and every event would fail to
        parse -- or worse, parse into nonsense.
        """
        rows = [("2013-01-27 21:42:38", "51549", "iJTBIGHPjgJcT4Bt")] * 8
        path = write_tsv(
            tmp_path / "events.tsv",
            rows,
            header=["timestamp", "user_id", "track_id"],
        )
        schema = sniff_schema(path)

        assert schema.ts_idx == 0
        assert schema.ts_resolved_by == "header name"
        assert schema.ts_kind == "iso"
        assert schema.user_idx == 1
        assert schema.track_idx == 2

    def test_real_corpus_datetime_format_parses_to_the_right_instant(self, tmp_path):
        """The format the actual archive uses, end to end."""
        rows = [
            ("51549", "iJTBIGHPjgJcT4Bt", "2013-01-27 21:42:38"),
            ("51549", "iJTBIGHPjgJcT4Bt", "2013-01-27 21:38:53"),
        ]
        path = write_tsv(tmp_path / "events.tsv", rows, header=["user_id", "track_id", "timestamp"])
        corpus = build(path, sniff_schema(path))

        assert corpus.ts.tolist() == [1359322733, 1359322958]
        assert datetime.fromtimestamp(int(corpus.ts[0]), UTC).isoformat() == (
            "2013-01-27T21:38:53+00:00"
        )

    def test_millisecond_timestamps_are_recognised(self, tmp_path):
        rows = [(f"user{i % 2}", f"track{i}", (BASE_TS + i) * 1000) for i in range(8)]
        path = write_tsv(tmp_path / "events.tsv", rows)

        assert sniff_schema(path).ts_kind == "millis"

    def test_implausible_numeric_timestamp_is_refused(self, tmp_path):
        """A bare counter is not a timestamp; guessing a unit here is how you
        end up with an entire corpus dated 1970."""
        rows = [(f"user{i}", f"track{i}", i) for i in range(8)]
        path = write_tsv(tmp_path / "events.tsv", rows)

        with pytest.raises(ValueError, match="neither plausible unix seconds"):
            sniff_schema(path)

    def test_reads_through_bz2(self, tmp_path):
        path = write_tsv(tmp_path / "events.tsv.bz2", simple_rows(), compress=True)
        schema = sniff_schema(path)

        assert schema.ts_kind == "seconds"
        assert schema.has_header is False


class TestBuild:
    def test_counts_and_vocabularies(self, tmp_path):
        rows = simple_rows(n_users=3, n_tracks=4, plays=2)
        path = write_tsv(tmp_path / "events.tsv", rows)
        corpus = build(path, sniff_schema(path))

        assert corpus.n_events == len(rows) == 24
        assert corpus.n_users == 3
        assert corpus.n_tracks == 4
        corpus.validate()

    def test_sorted_by_user_then_time_from_scrambled_input(self, tmp_path):
        """The real archive's ordering is not documented, so the build must not
        depend on it.

        Compared per user rather than array-to-array: codes are assigned in
        first-appearance order, so reversed input numbers the users backwards
        and the blocks come out in a different order. What must be identical is
        the history each original user id resolves to.
        """
        rows = simple_rows()
        forward = write_tsv(tmp_path / "a.tsv", rows)
        reversed_ = write_tsv(tmp_path / "b.tsv", list(reversed(rows)))

        a = build(forward, sniff_schema(forward))
        b = build(reversed_, sniff_schema(reversed_))

        def histories(corpus):
            out = {}
            for code, user in enumerate(corpus.users.tolist()):
                window = corpus.events_for(code)
                out[user] = [
                    (corpus.tracks[t], int(ts))
                    for t, ts in zip(corpus.track_code[window], corpus.ts[window], strict=True)
                ]
            return out

        assert histories(a) == histories(b)
        assert sorted(a.users.tolist()) == sorted(b.users.tolist())
        b.validate()

    def test_user_offsets_bound_each_users_events(self, tmp_path):
        path = write_tsv(tmp_path / "events.tsv", simple_rows(n_users=4, n_tracks=3, plays=2))
        corpus = build(path, sniff_schema(path))

        for user in range(corpus.n_users):
            window = corpus.events_for(user)
            assert np.all(corpus.user_code[window] == user)
            assert np.all(np.diff(corpus.ts[window]) >= 0)
        assert int(corpus.user_offsets[-1]) == corpus.n_events

    def test_millisecond_input_lands_in_the_right_decade(self, tmp_path):
        """The regression this exists for: dividing by an assumed resolution
        puts every event in 1970 and nothing raises."""
        rows = [(f"user{i % 2}", f"track{i % 3}", (BASE_TS + i * 60) * 1000) for i in range(12)]
        path = write_tsv(tmp_path / "events.tsv", rows)
        corpus = build(path, sniff_schema(path))

        assert datetime.fromtimestamp(int(corpus.ts.min()), UTC).year == 2013

    def test_grows_past_an_underestimated_capacity(self, tmp_path):
        rows = simple_rows(n_users=5, n_tracks=5, plays=3)
        path = write_tsv(tmp_path / "events.tsv", rows)
        corpus = build(path, sniff_schema(path), capacity=4)

        assert corpus.n_events == len(rows)
        corpus.validate()

    def test_estimate_rows_has_a_floor(self, tmp_path):
        path = write_tsv(tmp_path / "events.tsv", simple_rows())

        assert estimate_rows(path) >= 1 << 16


class TestRoundTrip:
    def test_save_and_load_are_identical(self, tmp_path):
        path = write_tsv(tmp_path / "events.tsv", simple_rows())
        built = build(path, sniff_schema(path))
        built.save(tmp_path / "store")

        loaded = CompactCorpus.load(tmp_path / "store", mmap=False)

        assert np.array_equal(built.user_code, loaded.user_code)
        assert np.array_equal(built.track_code, loaded.track_code)
        assert np.array_equal(built.ts, loaded.ts)
        assert np.array_equal(built.user_offsets, loaded.user_offsets)
        assert built.users.tolist() == loaded.users.tolist()
        loaded.validate()

    def test_build_is_deterministic(self, tmp_path):
        path = write_tsv(tmp_path / "events.tsv", simple_rows())
        schema = sniff_schema(path)

        first, second = build(path, schema), build(path, schema)

        assert np.array_equal(first.user_code, second.user_code)
        assert np.array_equal(first.track_code, second.track_code)
        assert np.array_equal(first.ts, second.ts)


class TestValidate:
    def _corpus(self):
        return CompactCorpus(
            user_code=np.array([0, 0, 1, 1], dtype=np.int32),
            track_code=np.array([0, 1, 0, 1], dtype=np.int32),
            ts=np.array([10, 20, 30, 40], dtype=np.int32),
            user_offsets=np.array([0, 2, 4], dtype=np.int64),
            users=np.array(["a", "b"]),
            tracks=np.array(["t0", "t1"]),
        )

    def test_accepts_a_well_formed_corpus(self):
        self._corpus().validate()

    def test_rejects_unsorted_time_within_a_user(self):
        corpus = self._corpus()
        corpus.ts = np.array([20, 10, 30, 40], dtype=np.int32)

        with pytest.raises(ValueError, match="not sorted by ts"):
            corpus.validate()

    def test_allows_time_to_go_backwards_across_a_user_boundary(self):
        """Each user's history starts whenever it starts. A global monotonicity
        check would reject every real corpus."""
        corpus = self._corpus()
        corpus.ts = np.array([100, 200, 10, 20], dtype=np.int32)

        corpus.validate()

    def test_rejects_events_not_grouped_by_user(self):
        corpus = self._corpus()
        corpus.user_code = np.array([0, 1, 0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="not grouped by user"):
            corpus.validate()

    def test_rejects_offsets_that_do_not_span_the_arrays(self):
        corpus = self._corpus()
        corpus.user_offsets = np.array([0, 2, 3], dtype=np.int64)

        with pytest.raises(ValueError, match="span exactly"):
            corpus.validate()


class TestStats:
    def test_unbounded_recurrence_rate_is_the_share_of_pairs_played_twice(self, tmp_path):
        # 2 users x 3 tracks. Every pair is played twice except (user1, track2),
        # played once: 5 of 6 pairs recur.
        rows = []
        ts = BASE_TS
        for u in range(2):
            for t in range(3):
                plays = 1 if (u, t) == (1, 2) else 2
                for _ in range(plays):
                    rows.append((f"user{u}", f"track{t}", ts))
                    ts += 60
        path = write_tsv(tmp_path / "events.tsv", rows)
        stats = corpus_stats(build(path, sniff_schema(path)), min_track_plays=1)

        assert stats["pairs"]["measured"] == 6
        assert stats["plays_per_pair"]["recurring_pairs"] == 5
        assert stats["plays_per_pair"]["unbounded_recurrence_rate"] == pytest.approx(
            5 / 6, abs=1e-4
        )

    def test_catalog_cutoff_drops_tracks_below_the_threshold(self, tmp_path):
        # track0 gets 4 plays, track1 gets 1.
        rows = [("userA", "track0", BASE_TS + i * 60) for i in range(4)]
        rows.append(("userA", "track1", BASE_TS + 999))
        path = write_tsv(tmp_path / "events.tsv", rows)
        stats = corpus_stats(build(path, sniff_schema(path)), min_track_plays=2)

        assert stats["catalog_cutoff"]["tracks_kept"] == 1
        assert stats["catalog_cutoff"]["tracks_dropped"] == 1
        assert stats["catalog_cutoff"]["events_kept"] == 4

    def test_active_span_is_measured_per_user_in_days(self, tmp_path):
        rows = [
            ("userA", "t0", BASE_TS),
            ("userA", "t1", BASE_TS + 10 * 86400),
            ("userB", "t0", BASE_TS),
            ("userB", "t1", BASE_TS + 40 * 86400),
        ]
        path = write_tsv(tmp_path / "events.tsv", rows)
        stats = corpus_stats(build(path, sniff_schema(path)), min_track_plays=1)

        assert stats["active_span_days_per_user"]["at_least_30d"] == 1
        assert stats["span"]["days"] == pytest.approx(40.0, abs=0.1)


class TestCountsFile:
    def test_totals_match_a_synthesized_counts_file(self, tmp_path):
        rows = [("userA", "t0", 3), ("userA", "t1", 1), ("userB", "t0", 7)]
        path = write_tsv(tmp_path / "counts.tsv", rows)

        totals = read_counts_totals(path)

        assert totals["pairs"] == 3
        assert totals["plays"] == 11

    def test_header_row_is_skipped(self, tmp_path):
        """The real counts file carries `user_id  track_id  count`.

        Without skipping it pyarrow tries to read the literal string "count" as
        an int64 and the whole cross-check dies -- which is exactly what it did
        on the first full build.
        """
        rows = [("userA", "t0", 3), ("userB", "t0", 7)]
        path = write_tsv(tmp_path / "counts.tsv", rows, header=["user_id", "track_id", "count"])

        totals = read_counts_totals(path)

        assert totals["pairs"] == 2
        assert totals["plays"] == 10

    def test_headerless_counts_file_is_still_read_whole(self, tmp_path):
        """Skipping a row that is not a header would lose a real pair."""
        rows = [("userA", "t0", 3), ("userB", "t0", 7)]
        path = write_tsv(tmp_path / "counts.tsv", rows)

        assert read_counts_totals(path)["pairs"] == 2

    def test_counts_agree_with_the_events_file_they_summarize(self, tmp_path):
        """The cross-check the real build performs, in miniature: the counts
        file's pair count is the encounter count, and its sum is the events."""
        events = simple_rows(n_users=2, n_tracks=3, plays=2)
        events_path = write_tsv(tmp_path / "events.tsv", events)
        corpus = build(events_path, sniff_schema(events_path))
        stats = corpus_stats(corpus, min_track_plays=1)

        counts = [(f"user{u}", f"track{t}", 2) for u in range(2) for t in range(3)]
        counts_path = write_tsv(tmp_path / "counts.tsv", counts)
        totals = read_counts_totals(counts_path)

        assert totals["pairs"] == stats["pairs"]["measured"]
        assert totals["plays"] == stats["events"]["measured"]
