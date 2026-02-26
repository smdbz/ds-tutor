# ds-tutor

A scikit-learn wrapper to help automate pipeline diagnostics.




## Execution Flow: What happens when you run a cell

When a user builds a `TutorPipeline` and hits **Shift+Enter** to execute `pipeline.fit(X, y)` in a Jupyter Notebook, the following logical steps occur:

1. **Interception**: `TutorPipeline.fit(X, y)` intercepts the standard scikit-learn `fit` method.
2. **Data Validation**: The pipeline checks if the input `X` is a pandas `DataFrame`. If it is not, the pipeline immediately falls back to the standard `sklearn` behavior.
3. **Tutor Evaluation**: If `X` is a `DataFrame`, the pipeline invokes its internal `_run_tutors(X)` method, iterating sequentially through all registered tutors (e.g., `ImputerTutor`, `ScalingTutor`).
4. **Condition Checking**: For each tutor, `tutor.check_condition(X)` is evaluated against the dataset. This triggers specific heuristic checks (e.g., "Are there missing values?", "Is the variance highly imbalanced?").
5. **Insights Rendering**: If a tutor's condition returns `True`, `IPython.display` is used to gracefully inject markdown and interactive visual insights directly into the notebook cell output:
   - **Reasoning**: A plain English explanation of the issue (`get_reasoning()`).
   - **Math**: The underlying LaTeX mathematical justification (`get_math()`).
   - **Code Suggestions**: A copy-pasteable scikit-learn Python snippet to resolve the issue (`get_code_suggestion()`).
   - **Visuals**: An interactive Plotly/Matplotlib figure illustrating the problem in the data (`get_visual(X)`).
6. **Standard Execution**: After all tutors have been evaluated and their insights are displayed on the screen, the wrapper calls `super().fit(X, y)` to execute the actual standard scikit-learn model fitting.
7. **Failure Context**: If the underlying model naturally fails (e.g., `LinearRegression` encounters `NaN`s), the standard traceback is printed *after* the tutor's advice, giving the user immediate context on why the failure occurred and the exact code needed to fix it.