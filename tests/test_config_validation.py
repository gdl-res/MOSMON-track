from mosmon_tracking.config import find_unknown_keys


def test_clean_config_has_no_unknown_keys():
    data = {"model": {"imgsz": 1920, "conf": 0.15}, "video": {"frame_stride": 2}}
    assert find_unknown_keys(data) == []


def test_top_level_typo_is_flagged():
    assert find_unknown_keys({"veideo": {}}) == ["veideo"]


def test_nested_typo_is_flagged_with_dotted_path():
    # a realistic typo: missing the 'c' in 'tracking'
    unknown = find_unknown_keys({"video": {"save_traking_video": True}})
    assert unknown == ["video.save_traking_video"]


def test_multiple_and_mixed():
    data = {"video": {"frame_stride": 2, "bogus": 1}, "nope": {}}
    assert set(find_unknown_keys(data)) == {"video.bogus", "nope"}
