from html import escape

from IPython.display import HTML, display
from sklearn import set_config
from rich.console import Console
from rich.markdown import Markdown


def _eda_theme() -> str:
    return """
<style>
.dstutor-shell {
  font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
  margin: 8px 0 14px 0;
}
.dstutor-head {
  background: linear-gradient(110deg, #f3f9f8 0%, #edf4fb 100%);
  border: 1px solid #d6e6e4;
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.dstutor-title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: #11383b;
}
.dstutor-sub {
  margin-top: 4px;
  font-size: 12px;
  color: #4e6767;
}
.dstutor-card {
  border: 1px solid #dde7ef;
  border-radius: 10px;
  padding: 12px 14px;
  margin: 10px 0;
  background: #ffffff;
}
.dstutor-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.dstutor-card-title {
  font-size: 15px;
  font-weight: 650;
  color: #1f2b36;
}
.dstutor-badge {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .03em;
  padding: 3px 8px;
  border-radius: 999px;
  text-transform: uppercase;
}
.dstutor-badge-hit {
  color: #0e624f;
  background: #def7ef;
  border: 1px solid #bfeede;
}
.dstutor-badge-skip {
  color: #5f6d79;
  background: #f1f4f8;
  border: 1px solid #e4e9ef;
}
.dstutor-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .03em;
  color: #55697a;
}
.dstutor-text {
  margin: 4px 0 8px 0;
  color: #243240;
  font-size: 13px;
}
</style>
"""


def _render_eda_card(result) -> str:
    badge_class = "dstutor-badge-hit" if result.triggered else "dstutor-badge-skip"
    badge_text = "Applied" if result.triggered else "No Action"
    return f"""
<div class="dstutor-card">
  <div class="dstutor-row">
    <div class="dstutor-card-title">{result.title}</div>
    <span class="dstutor-badge {badge_class}">{badge_text}</span>
  </div>
  <div class="dstutor-label">Diagnosis</div>
  <div class="dstutor-text">{result.diagnosis}</div>
  <div class="dstutor-label">Action</div>
  <div class="dstutor-text">{result.action}</div>
</div>
"""


def render_experiment_summary(experiment):
    """Build and display a Jupyter tab UI for an Experiment."""
    try:
        import ipywidgets as widgets
    except ImportError:
        print("Install `ipywidgets` to render experiment.summary() tabs.")
        return

    out_ledger = widgets.Output()
    out_theory = widgets.Output()
    out_pipeline = widgets.Output()

    with out_ledger:
        display(experiment.leaderboard())

    with out_theory:
        console = Console(force_jupyter=True)
        if experiment.theory_log:
            markdown = "\n\n".join(experiment.theory_log)
            console.print(Markdown(markdown))
        else:
            console.print(Markdown("No tutor notes recorded."))

    with out_pipeline:
        set_config(display="diagram")
        if experiment.best_pipeline is not None:
            display(experiment.best_pipeline)
        else:
            print("Run experiment.run() to compile and fit the best pipeline.")

    tabs = widgets.Tab(children=[out_ledger, out_theory, out_pipeline])
    tabs.set_title(0, "Results Ledger")
    tabs.set_title(1, "Theory & Actions")
    tabs.set_title(2, "Best Pipeline")
    display(tabs)


def _render_theory_block(result) -> str:
    status = "Applied" if result.triggered else "Reviewed"
    return f"""
<section class="dstutor-card">
  <div class="dstutor-row">
    <div class="dstutor-card-title">{escape(result.title)} Theory</div>
    <span class="dstutor-badge {'dstutor-badge-hit' if result.triggered else 'dstutor-badge-skip'}">{status}</span>
  </div>
  <div class="dstutor-text" style="white-space: pre-wrap;">{escape(result.theory)}</div>
</section>
"""


def _render_tab_shell(diagnostics_html: str, theory_html: str) -> str:
    return f"""
{_eda_theme()}
<style>
.dstutor-tabs-wrap {{
  font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
}}
.dstutor-tabs {{
  display: flex;
  flex-wrap: wrap;
  margin-top: 8px;
}}
.dstutor-tabs input[type="radio"] {{
  position: absolute;
  opacity: 0;
}}
.dstutor-tabs label {{
  order: 1;
  padding: 8px 12px;
  margin-right: 8px;
  margin-bottom: 8px;
  border: 1px solid #d5e3ec;
  border-radius: 9px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  color: #3e5364;
  background: #f7fbff;
}}
.dstutor-tabs .tab-content {{
  order: 99;
  width: 100%;
  border: 1px solid #d5e3ec;
  border-radius: 12px;
  padding: 12px 14px;
  background: #ffffff;
  display: none;
}}
.dstutor-tabs input[type="radio"]:checked + label {{
  color: #13404f;
  background: #e9f5f8;
  border-color: #b9dce5;
}}
#dstutor-tab-eda:checked ~ #dstutor-content-eda,
#dstutor-tab-theory:checked ~ #dstutor-content-theory {{
  display: block;
}}
.dstutor-detail table {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 8px;
  margin-bottom: 10px;
  font-size: 12px;
}}
.dstutor-detail th, .dstutor-detail td {{
  border: 1px solid #e6edf3;
  padding: 6px 8px;
  text-align: left;
}}
.dstutor-detail th {{
  background: #f5f9fc;
}}
</style>
<section class="dstutor-shell dstutor-tabs-wrap">
  <div class="dstutor-tabs">
    <input type="radio" id="dstutor-tab-eda" name="dstutor-tab" checked>
    <label for="dstutor-tab-eda">EDA Diagnostics</label>
    <input type="radio" id="dstutor-tab-theory" name="dstutor-tab">
    <label for="dstutor-tab-theory">Theory</label>
    <div id="dstutor-content-eda" class="tab-content">{diagnostics_html}</div>
    <div id="dstutor-content-theory" class="tab-content">{theory_html}</div>
  </div>
</section>
"""


def render_eda_tutor_summary(eda_tutor):
    """Build and display a tabbed EDA tutor summary for Jupyter."""

    results = eda_tutor.last_results
    if not results:
        results = eda_tutor.teach_me()

    triggered_count = sum(1 for result in results if result.triggered)
    project_name = eda_tutor.context.name
    experiment_name = eda_tutor.config.name

    diagnostics_parts = [
        f"""
<section class="dstutor-shell">
  <div class="dstutor-head">
    <h3 class="dstutor-title">EDA Tutor Summary</h3>
    <div class="dstutor-sub">
      Project: <strong>{escape(project_name)}</strong> |
      Experiment: <strong>{escape(experiment_name)}</strong> |
      Checks triggered: <strong>{triggered_count}/{len(results)}</strong>
    </div>
  </div>
</section>
"""
    ]

    for result in results:
        diagnostics_parts.append(_render_eda_card(result))
        if result.details is not None and not result.details.empty:
            detail_df = result.details.copy()
            if "missing_pct" in detail_df.columns:
                detail_df["missing_pct"] = detail_df["missing_pct"].map(lambda x: f"{x:.2f}%")
            if "abs_skew" in detail_df.columns:
                detail_df["abs_skew"] = detail_df["abs_skew"].map(lambda x: f"{x:.3f}")
            diagnostics_parts.append(
                f'<div class="dstutor-detail">{detail_df.to_html(index=False, border=0, escape=True)}</div>'
            )

    theory_parts = [_render_theory_block(result) for result in results]
    if not theory_parts:
        theory_parts = ['<section class="dstutor-card"><div class="dstutor-text">No tutor output available.</div></section>']

    display(HTML(_render_tab_shell("".join(diagnostics_parts), "".join(theory_parts))))
