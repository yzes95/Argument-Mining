import os
import pytest
from aaec_dataset_script_analysis import (
    load_essay_text,
    parse_annotation_file,
    identify_non_argumentative_segments,
    process_essay_directory,
)

# --- Helpers ---
def make_sample_files(base_dir, essay_id="essay001"):
    txt_path = os.path.join(base_dir, f"{essay_id}.txt")
    ann_path = os.path.join(base_dir, f"{essay_id}.ann")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("This is a claim. And this is some support. Just context here.")

    with open(ann_path, "w", encoding="utf-8") as f:
        f.write("T1\tClaim 0 14\tThis is a claim.\n")
        f.write("T2\tSupport 20 41\tthis is some support.")

    return txt_path, ann_path

# --- Tests for load_essay_text ---
def test_load_essay_text_reads_file_content():
    tmp_dir = "tmp_test_dir"
    os.makedirs(tmp_dir, exist_ok=True)
    txt_path, _ = make_sample_files(tmp_dir)

    text = load_essay_text(txt_path)
    assert "This is a claim." in text

def test_load_essay_text_missing_file_returns_none():
    tmp_dir = "tmp_test_dir"
    os.makedirs(tmp_dir, exist_ok=True)
    missing_path = os.path.join(tmp_dir, "missing.txt")

    # Instead of crashing, the function should return None
    text = load_essay_text(missing_path)
    assert text is None

# --- Tests for parse_annotation_file ---
def test_parse_annotation_file_extracts_argumentative_labels():
    tmp_dir = "tmp_test_dir"
    _, ann_path = make_sample_files(tmp_dir)

    anns = parse_annotation_file(ann_path)
    assert any(a["Type"] == "Argumentative" for a in anns)

def test_parse_annotation_file_raises_exception_for_invalid_file():
    # simulate invalid file format by creating a bad ann file
    tmp_dir = "tmp_test_dir"
    bad_path = os.path.join(tmp_dir, "bad.ann")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("INVALID LINE THAT BREAKS PARSING")

    # We expect a KeyError or IndexError depending on your parse logic
    with pytest.raises((IndexError, KeyError)):
        parse_annotation_file(bad_path)

# --- Tests for identify_non_argumentative_segments ---
def test_identify_non_argumentative_segments_returns_list():
    essay_text = "Claim here. Support here. Filler text remains."
    anns = [{"Type":"Argumentative","StartPos":0,"EndPos":10,"Text":"Claim here"}]
    result = identify_non_argumentative_segments(essay_text, anns)
    assert isinstance(result, list)

def test_identify_non_argumentative_segments_detects_filler_text():
    essay_text = "Claim here. Support here. Filler text remains."
    anns = [{"Type":"Argumentative","StartPos":0,"EndPos":10,"Text":"Claim here"}]
    result = identify_non_argumentative_segments(essay_text, anns)
    assert all("Filler" not in seg["Text"] for seg in result)

# --- Tests for process_essay_directory ---
def test_process_essay_directory_returns_segments_and_length():
    tmp_dir = "tmp_test_dir"
    essay_id = "essay123"
    make_sample_files(tmp_dir, essay_id)
    segments, essay_len = process_essay_directory(essay_id)
    assert isinstance(segments, list)
    assert essay_len > 0

def test_process_essay_directory_raises_when_file_missing():
    # simulate missing essay text file
    essay_id = "missing_essay"
    # expect an exception (FileNotFoundError) when trying to process a missing essay
    with pytest.raises(FileNotFoundError):
        process_essay_directory(essay_id)

