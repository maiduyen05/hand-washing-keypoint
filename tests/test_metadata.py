"""Metadata tests."""

from src.dataset_metadata import participant_id_from_filename


def test_regular_participant_filename() -> None:
    assert participant_id_from_filename("2021-08-13_13-25-36-76-2.mp4", 2) == 76


def test_explicit_filename_exceptions() -> None:
    assert participant_id_from_filename("2021-06-30_09-56-19-11labots.mp4", 1) == 1
    assert participant_id_from_filename("2021-06-30_09-50-42-13.mp4", 3) == 1


def test_malformed_suffix_still_uses_participant_prefix() -> None:
    assert participant_id_from_filename("2021-08-12_11-07-14-62-22.mp4", 2) == 62
    assert participant_id_from_filename("2021-08-12_13-29-59-64+-2.mp4", 2) == 64
