def test_input_dimension_matches(load_sample_data, load_model):
    X = load_sample_data.X
    model = load_model

    # sklearn models usually have n_features_in_ attribute
    if hasattr(model, "n_features_in_"):
        assert X.shape[1] == model.n_features_in_, \
            f"Input dimension {X.shape[1]} does not match model {model.n_features_in_}"
