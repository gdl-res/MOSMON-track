from mosmon_tracking.metadata import (
    SPECIES_AEGYPTI,
    SPECIES_ALBOPICTUS,
    SPECIES_PIPIENS,
    SPECIES_STEPHENSI,
    parse_filename,
    resolve_species_fragment,
)


def test_full_single_species():
    name = ("20251007_mosmon_iss_aedes_albopictus_stage3and4_djiosmoaction5_video_"
            "4k30fps_rocksteadywide_lightartwarm_camerapositionA_stdcontainer_depth45mm.MP4")
    m = parse_filename(name)
    assert m.recording_date == "2025-10-07"
    assert m.species == [SPECIES_ALBOPICTUS]
    assert m.stage == "3-4"
    assert m.camera_family == "dji"
    assert m.fps == 30
    assert m.resolution == "4K"
    assert m.camera_position == "A"
    assert m.container_type == "std"
    assert m.water_depth_mm == 45.0
    assert "artificial_warm" in m.lighting
    assert m.modality == "video"


def test_dual_container_sx_dx():
    name = ("20251009_alboSX_culexDX_stage3and4_djiosmoaction5_video_4k30fps_"
            "rocksteadywide_lightnatartcold_cp4_big2boxcontainer_depth45mm_larvaeadded.MP4")
    m = parse_filename(name)
    assert m.is_dual_container is True
    assert m.species_left == SPECIES_ALBOPICTUS
    assert m.species_right == SPECIES_PIPIENS
    assert set(m.species) == {SPECIES_ALBOPICTUS, SPECIES_PIPIENS}
    assert m.camera_position == "4"
    assert m.container_type == "big2box"
    assert "larvaeadded" in m.modifiers
    assert "natural" in m.lighting and "artificial_cold" in m.lighting


def test_typo_lighting_and_abbreviations():
    name = ("20251008_mosmon_iss_aedes_aegypti_stage3and4_insta360acepro2_video_4k60fps_"
            "distlinear_lightartcold_camposB_stdcleancontainer_depth45mm_larvaeadd.mp4")
    m = parse_filename(name)
    assert m.species == [SPECIES_AEGYPTI]
    assert m.camera_family == "insta360"
    assert m.fps == 60
    assert m.camera_position == "B"
    assert m.container_type == "std_clean"
    assert "artificial_cold" in m.lighting


def test_resolution_dot_notation_and_stephensi():
    name = ("20251010_mosmon_iss_anopheles_stephensis_stage3_goprohero13black_video_"
            "5dot7Kwidescreen_distlinear_lightnatural_cpfree_fountain.MP4")
    m = parse_filename(name)
    assert m.species == [SPECIES_STEPHENSI]
    assert m.stage == "3"
    assert m.resolution == "5.7K"
    assert m.camera_position == "FREE"
    assert m.container_type == "fountain"


def test_missing_tokens_are_robust():
    m = parse_filename("randomclip.mp4")
    assert m.recording_date is None
    assert m.species == []
    assert any("date" in w for w in m.warnings)
    assert any("species" in w for w in m.warnings)


def test_resolve_species_fragment():
    assert resolve_species_fragment("albopictussx"[:-2]) == SPECIES_ALBOPICTUS
    assert resolve_species_fragment("aegypti") == SPECIES_AEGYPTI
    assert resolve_species_fragment("culex") == SPECIES_PIPIENS
    assert resolve_species_fragment("xyz") is None
