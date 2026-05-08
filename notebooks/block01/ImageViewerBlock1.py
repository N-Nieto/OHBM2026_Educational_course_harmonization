from __future__ import annotations

from pathlib import Path
from shutil import copy2
from urllib.error import URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError as exc:  # pragma: no cover - depends on notebook environment
    raise ImportError(
        "ipywidgets is required for the MRI viewer. Install it with "
        "`pip install ipywidgets`."
    ) from exc

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "nibabel is required to load `.nii` and `.nii.gz` MRI files. Install it "
        "with `pip install nibabel`."
    ) from exc


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_SAMPLE_PATH = DATA_DIR / "example4d.nii.gz"
DEFAULT_SAMPLE_URL = (
    "https://afni.nimh.nih.gov/pub/dist/bin/linux_fedora_25_64/"
    "meica.libs/nibabel/tests/data/example4d.nii.gz"
)
VIEW_TO_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}


def _is_valid_nifti(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        nib.load(str(path))
    except Exception:
        return False

    return True


def _bundled_sample_mri_path() -> Path | None:
    try:
        from nibabel.testing import data_path
    except ImportError:  # pragma: no cover - depends on nibabel installation
        return None

    candidate = Path(data_path) / "example4d.nii.gz"
    return candidate if candidate.exists() else None


def download_sample_mri(
    destination: Path | str = DEFAULT_SAMPLE_PATH,
    url: str = DEFAULT_SAMPLE_URL,
) -> Path:
    """Prepare a small sample MRI volume for the viewer."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if _is_valid_nifti(destination):
        return destination

    bundled_sample = _bundled_sample_mri_path()
    if bundled_sample is not None:
        copy2(bundled_sample, destination)
        return destination

    try:
        urlretrieve(url, destination)
    except URLError as exc:  # pragma: no cover - requires network access
        raise RuntimeError(
            "Unable to download the sample MRI file. Check your internet "
            f"connection or provide a local `.nii.gz` file instead. URL: {url}"
        ) from exc

    return destination


def load_mri_volume(image_path: Path | str) -> np.ndarray:
    """Load a NIfTI image and return a 3D array ready for visualization."""

    image = nib.load(str(image_path))
    data = np.asarray(image.get_fdata())

    if data.ndim == 4:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D or 4D MRI volume, but found an array with shape {data.shape}."
        )

    data = np.nan_to_num(data, copy=False)
    # Rotate 180 degrees to match the orientation of the sample image]
    return data



class MRIImageViewer:
    """Interactive MRI slice viewer for Jupyter notebooks."""

    def __init__(
        self,
        image_path: Path | str,
        second_image_path: Path | str | None = None,
        *,
        cmap: str = "gray",
        figsize: tuple[int, int] = (6, 6),
    ) -> None:
        self.image_paths = [Path(image_path)]
        if second_image_path is not None:
            self.image_paths.append(Path(second_image_path))

        self.volumes = [load_mri_volume(path) for path in self.image_paths]
        self.cmap = cmap
        self.figsize = figsize
        self.current_view = "axial"

        self.output = widgets.Output()
        self.status = widgets.HTML()

        self.axial_button = widgets.Button(description="Axial", button_style="primary")
        self.coronal_button = widgets.Button(description="Coronal")
        self.sagittal_button = widgets.Button(description="Sagittal")
        self.prev_button = widgets.Button(description="Previous Slice")
        self.next_button = widgets.Button(description="Next Slice")

        self.slice_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description="Slice",
            continuous_update=False,
        )

        self._connect_events()
        self._set_view("axial")

    def _connect_events(self) -> None:
        self.axial_button.on_click(lambda _: self._set_view("axial"))
        self.coronal_button.on_click(lambda _: self._set_view("coronal"))
        self.sagittal_button.on_click(lambda _: self._set_view("sagittal"))
        self.prev_button.on_click(lambda _: self._step_slice(-1))
        self.next_button.on_click(lambda _: self._step_slice(1))
        self.slice_slider.observe(self._on_slice_change, names="value")

    def _set_view(self, view: str) -> None:
        self.current_view = view
        max_slice = self._get_slider_max(view)
        middle_slice = max_slice // 2

        self.slice_slider.max = max_slice
        self.slice_slider.description = f"{view.title()} Slice"
        self.slice_slider.value = middle_slice
        self._update_view_buttons()
        self._render()

    def _update_view_buttons(self) -> None:
        styles = {
            "axial": self.axial_button,
            "coronal": self.coronal_button,
            "sagittal": self.sagittal_button,
        }

        for view, button in styles.items():
            button.button_style = "primary" if view == self.current_view else ""

    def _step_slice(self, delta: int) -> None:
        new_value = int(np.clip(self.slice_slider.value + delta, 0, self.slice_slider.max))
        self.slice_slider.value = new_value

    def _on_slice_change(self, change: dict) -> None:
        if change.get("name") == "value":
            self._render()

    def _get_slider_max(self, view: str) -> int:
        axis = VIEW_TO_AXIS[view]
        return max(volume.shape[axis] - 1 for volume in self.volumes)

    def _get_volume_slice_index(self, volume: np.ndarray) -> int:
        axis = VIEW_TO_AXIS[self.current_view]
        volume_max_slice = volume.shape[axis] - 1

        if self.slice_slider.max == 0 or volume_max_slice == 0:
            return 0

        normalized_position = self.slice_slider.value / self.slice_slider.max
        return int(round(normalized_position * volume_max_slice))

    def _get_slice(self, volume: np.ndarray) -> tuple[np.ndarray, int, int]:
        slice_index = self._get_volume_slice_index(volume)

        if self.current_view == "axial":
            slice_data = volume[:, :, slice_index]
        elif self.current_view == "coronal":
            slice_data = volume[:, slice_index, :]
        else:
            slice_data = volume[slice_index, :, :]


        return np.rot90(slice_data,k=3,axes = (0,1)), slice_index, volume.shape[VIEW_TO_AXIS[self.current_view]]

    def _render(self) -> None:
        slice_views = [self._get_slice(volume) for volume in self.volumes]
        ncols = len(slice_views)

        with self.output:
            self.output.clear_output(wait=True)
            figure_size = self.figsize
            if ncols > 1 and figure_size == (6, 6):
                figure_size = (12, 6)

            fig, axes = plt.subplots(1, ncols, figsize=figure_size, squeeze=False)
            for ax, (path, (slice_data, slice_index, total_slices)) in zip(
                axes[0], zip(self.image_paths, slice_views)
            ):
                ax.imshow(slice_data, cmap=self.cmap, origin="lower")
                ax.set_title(
                    f"{path.name}\n"
                    f"{self.current_view.title()} view | slice {slice_index + 1}/{total_slices}"
                )
                ax.axis("off")

            fig.tight_layout()
            plt.show()
            plt.close(fig)

        file_list = ", ".join(path.name for path in self.image_paths)
        shape_list = ", ".join(str(volume.shape) for volume in self.volumes)
        self.status.value = (
            f"<b>Files:</b> {file_list} | "
            f"<b>Shapes:</b> {shape_list} | "
            f"<b>View:</b> {self.current_view.title()}"
        )

    def widget(self) -> widgets.VBox:
        view_controls = widgets.HBox(
            [self.axial_button, self.coronal_button, self.sagittal_button]
        )
        slice_controls = widgets.HBox([self.prev_button, self.slice_slider, self.next_button])
        return widgets.VBox([view_controls, slice_controls, self.status, self.output])


    def show(self) -> widgets.VBox:
        return self.widget()


def show_sample_viewer(
    destination: Path | str = DEFAULT_SAMPLE_PATH,
    url: str = DEFAULT_SAMPLE_URL,
) -> MRIImageViewer:
    """Download a demo MRI volume if needed and display the interactive viewer."""

    image_path = download_sample_mri(destination=destination, url=url)
    viewer = MRIImageViewer(image_path)
    viewer.show()
    return viewer
