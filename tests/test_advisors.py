from unittest.mock import patch

import numpy as np
import pandas as pd

from ds_tutor import DsTutor


def test_dstutor_imputer_and_scaling():
    # Create dataset that needs imputation and scaling
    # Large scale difference triggers ScalingTutor
    # np.nan triggers ImputerTutor
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [100000.0, 200000.0, 300000.0, 400000.0, 500000.0]
    })
    y = [0, 1, 0, 1, 0]

    tutor = DsTutor()

    with patch('ds_tutor.core.display') as mock_display:
        # Prevent visual.show() from trying to open a browser window during tests
        with patch('plotly.graph_objs.Figure.show') as mock_show:
            tutor.evaluate(df)
            
            # Since df has missing values, ImputerTutor should have fired.
            # Since df has huge variance imbalance, ScalingTutor should have fired.
            assert mock_display.call_count > 0, "Expected IPython.display to be called"
            
            # Extract arguments from mock_display calls to verify the tutors ran
            displayed_texts = []
            for call in mock_display.call_args_list:
                args, kwargs = call
                if hasattr(args[0], 'data'):
                    displayed_texts.append(args[0].data)
                elif isinstance(args[0], str):
                    displayed_texts.append(args[0])
                    
            text_output = " ".join(displayed_texts)
            
            assert "Missing Data Detected." in text_output or "Missing Value Check" in text_output, "ImputerTutor did not run or output text correctly"
            assert "High Variance Imbalance Detected" in text_output or "Feature Scaling Check" in text_output, "ScalingTutor did not run or output text correctly"


def test_dstutor_no_issues():
    # Perfect dataset
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y = [0, 1, 0, 1, 0]

    tutor = DsTutor()

    with patch('ds_tutor.core.display') as mock_display:
        with patch('plotly.graph_objs.Figure.show') as mock_show:
            tutor.evaluate(df)
            
            # No tutors should have triggered
            assert mock_display.call_count == 0, "Expected no output for a perfect dataset"
