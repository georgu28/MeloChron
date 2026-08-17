"""Parser tests built on fixtures that reproduce the real files' hazards.

Written against synthesized files rather than the real corpora because the
lastfm-1K TSV is ~2.5 GB and the Spotify export has not arrived yet. Each
fixture encodes a specific documented failure mode.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from melochron import schema
from melochron.data import lastfm, spotify_export

# A track title containing a bare double quote, which is what breaks default
# pandas quoting and silently merges records.
_TSV_ROWS = [
    "user_000001\t2009-05-04T23:08:57Z\tmbid-a\tDeerhunter\tmbid-t\tAgoraphobia",
    "user_000001\t2009-05-04T23:12:01Z\t\tThe Fall\t\tHit the North",
    'user_000001\t2009-05-04T23:15:33Z\tmbid-b\tPavement\tmbid-u\tGold Soundz "live"',
    # Duplicate timestamp within a user, from a scrobble backfill.
    "user_000001\t2009-05-04T23:15:33Z\tmbid-c\tSlint\tmbid-v\tGood Morning, Captain",
    "user_000002\t2009-06-01T10:00:00Z\tmbid-d\tCan\tmbid-w\tVitamin C",
]


@pytest.fixture
def lastfm_tsv(tmp_path):
    p = tmp_path / lastfm.DEFAULT_FILENAME
    p.write_text("\n".join(_TSV_ROWS) + "\n", encoding="utf-8")
    return p


def test_bare_quotes_do_not_swallow_rows(lastfm_tsv):
    """The failure this guards against loses thousands of rows, not one."""
    df = lastfm.read_lastfm1k(lastfm_tsv)
    assert len(df) == len(_TSV_ROWS)
    assert 'Gold Soundz "live"' in set(df[schema.TRACK])


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2009-05-04T23:08:57Z", 1241478537),
        ("1970-01-01T00:00:01Z", 1),
        ("2024-03-19T22:26:38Z", 1710887198),
    ],
)
def test_to_unix_seconds_is_resolution_independent(raw, expected):
    """Pins the absolute value, which is the only way this bug shows up.

    ``parsed.astype("int64") // 1_000_000_000`` assumes nanosecond resolution.
    pandas 3.x hands back microseconds for many inputs, making that expression
    off by 1000x and landing every timestamp in 1970. Ordering within a single
    file still looks fine, so only an absolute assertion catches it.
    """
    assert int(schema.to_unix_seconds(pd.Series([raw])).iloc[0]) == expected


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_to_unix_seconds_marks_garbage_as_na():
    out = schema.to_unix_seconds(pd.Series(["not a date", "2024-03-19T22:26:38Z"]))
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1710887198


def test_lastfm_timestamps_are_absolute_correct(lastfm_tsv):
    df = lastfm.read_lastfm1k(lastfm_tsv)
    first = df[df[schema.USER] == "user_000001"].iloc[0]
    assert first[schema.TS] == 1241478537


def test_empty_mbids_are_not_errors(lastfm_tsv):
    df = lastfm.read_lastfm1k(lastfm_tsv)
    assert "The Fall" in set(df[schema.ARTIST])


def test_duplicate_timestamps_are_kept(lastfm_tsv):
    """Backfilled scrobbles share a timestamp; they are real consecutive plays."""
    df = lastfm.read_lastfm1k(lastfm_tsv)
    user = df[df[schema.USER] == "user_000001"]
    assert len(user) == 4
    assert user[schema.TS].duplicated().any()


def test_accepts_directory_and_conforms(lastfm_tsv):
    df = lastfm.read_lastfm1k(lastfm_tsv.parent)
    schema.validate(df)
    assert df[schema.SOURCE].unique().tolist() == [lastfm.SOURCE]
    # No play durations in this corpus; the column exists but is empty.
    assert df[schema.MS_PLAYED].isna().all()


def test_user_cap_yields_complete_histories(lastfm_tsv):
    df = lastfm.read_lastfm1k(lastfm_tsv, users=1)
    assert df[schema.USER].nunique() == 1
    assert len(df) == 4


def test_missing_file_names_the_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_lastfm1k"):
        lastfm.read_lastfm1k(tmp_path)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_extended_export_parses_and_drops_podcasts(tmp_path):
    _write_json(
        tmp_path / "Streaming_History_Audio_2024_1.json",
        [
            {
                "ts": "2024-03-19T22:26:38Z",
                "ms_played": 210_000,
                "master_metadata_track_name": "Weird Fishes",
                "master_metadata_album_artist_name": "Radiohead",
                "shuffle": False,
                "skipped": False,
                "offline": False,
            },
            {
                "ts": "2024-03-19T22:31:00Z",
                "ms_played": 900_000,
                "master_metadata_track_name": None,
                "master_metadata_album_artist_name": None,
                "episode_name": "Some Podcast Episode",
            },
        ],
    )

    df = spotify_export.read_export(tmp_path)
    schema.validate(df)
    assert len(df) == 1
    assert df[schema.TRACK].iloc[0] == "Weird Fishes"
    assert df[schema.MS_PLAYED].iloc[0] == 210_000
    assert df[schema.TS].iloc[0] == 1710887198
    assert df[schema.SOURCE].iloc[0] == spotify_export.SOURCE_EXTENDED


def test_account_export_converts_end_time_to_start(tmp_path):
    """``endTime`` marks the finish; the schema is start-ordered."""
    _write_json(
        tmp_path / "StreamingHistory_music_0.json",
        [
            {
                "endTime": "2024-03-19 22:26",
                "artistName": "Boards of Canada",
                "trackName": "Roygbiv",
                "msPlayed": 180_000,
            }
        ],
    )

    df = spotify_export.read_export(tmp_path)
    schema.validate(df)
    end = pd.Timestamp("2024-03-19 22:26", tz="UTC").timestamp()
    assert df[schema.TS].iloc[0] == int(end) - 180
    assert df[schema.SOURCE].iloc[0] == spotify_export.SOURCE_ACCOUNT


def test_extended_preferred_when_both_present(tmp_path):
    _write_json(
        tmp_path / "StreamingHistory_music_0.json",
        [{"endTime": "2024-03-19 22:26", "artistName": "A", "trackName": "B", "msPlayed": 1000}],
    )
    _write_json(
        tmp_path / "Streaming_History_Audio_2024_1.json",
        [
            {
                "ts": "2024-03-19T22:26:38Z",
                "ms_played": 210_000,
                "master_metadata_track_name": "C",
                "master_metadata_album_artist_name": "D",
            }
        ],
    )
    df = spotify_export.read_export(tmp_path)
    assert df[schema.SOURCE].iloc[0] == spotify_export.SOURCE_EXTENDED


def test_missing_export_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="no Spotify history JSON"):
        spotify_export.read_export(tmp_path)
