from scripts.reconstruct_pano_omega import model_args_from_checkpoint


def test_legacy_checkpoint_model_args_include_current_build_defaults() -> None:
    args = model_args_from_checkpoint({})

    assert args.enable_pano_geometry_residual is False
    assert args.aggregator_use_checkpoint is False
    assert args.dense_head_frames_chunk_size == 8
    assert args.dense_head_use_checkpoint is False
    assert args.dense_head_return_confidence is True


def test_checkpoint_model_args_preserve_current_build_values() -> None:
    args = model_args_from_checkpoint(
        {
            "enable_pano_geometry_residual": True,
            "aggregator_use_checkpoint": True,
            "dense_head_frames_chunk_size": 4,
            "dense_head_use_checkpoint": True,
            "dense_head_return_confidence": False,
        }
    )

    assert args.enable_pano_geometry_residual is True
    assert args.aggregator_use_checkpoint is True
    assert args.dense_head_frames_chunk_size == 4
    assert args.dense_head_use_checkpoint is True
    assert args.dense_head_return_confidence is False
