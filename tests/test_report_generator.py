from analyzers.report_generator import ReportGenerator


def test_report_generator_momentum_section():
    generator = ReportGenerator()
    
    correlation_result = {
        "correlation": 0.25,
        "p_value": 0.01,
        "significant": True,
    }
    
    section = generator.momentum_section(correlation_result)
    
    assert "0.25" in section
    assert "顯著" in section


def test_report_generator_feature_section():
    generator = ReportGenerator()
    
    feature_result = {
        "accuracy": 0.65,
        "feature_importance": {
            "RSI": 0.3,
            "MACD": 0.25,
            "SMA_20": 0.2,
        },
    }
    
    section = generator.feature_section(feature_result)
    
    assert "0.65" in section
    assert "RSI" in section
