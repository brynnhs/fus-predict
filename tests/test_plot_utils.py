import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fuspredict.plot_utils import savefig
from fuspredict.evaluation.visualization import savefig as viz_savefig


def _make_fig():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig


def test_savefig_writes_png_and_pdf(tmp_path):
    fig = _make_fig()
    savefig(fig, tmp_path / "myplot")
    assert (tmp_path / "myplot.png").exists()
    assert (tmp_path / "myplot.pdf").exists()


def test_savefig_accepts_suffixed_stem(tmp_path):
    fig = _make_fig()
    savefig(fig, tmp_path / "myplot.png")
    assert (tmp_path / "myplot.png").exists()
    assert (tmp_path / "myplot.pdf").exists()


def test_savefig_creates_parent_dirs(tmp_path):
    fig = _make_fig()
    savefig(fig, tmp_path / "nested" / "dir" / "myplot")
    assert (tmp_path / "nested" / "dir" / "myplot.png").exists()


def test_visualization_savefig_writes_png_and_pdf_and_closes(tmp_path):
    fig = _make_fig()
    n_open_before = len(plt.get_fignums())
    viz_savefig(fig, tmp_path / "report_fig")
    assert (tmp_path / "report_fig.png").exists()
    assert (tmp_path / "report_fig.pdf").exists()
    assert len(plt.get_fignums()) == n_open_before - 1
