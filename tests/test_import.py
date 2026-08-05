def test_package_imports():
    import geofarmai

    assert geofarmai.__version__ == "0.1.0"
    assert callable(geofarmai.run_pipeline)
    assert geofarmai.GeoFarmPipeline is not None
