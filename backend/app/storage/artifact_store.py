from pathlib import Path

import numpy as np
from PIL import Image


class ArtifactStore:
    def __init__(self, root: str = "../artifacts/images") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255).astype(np.uint8)

    def save_image(self, run_id: str, name: str, image_chw: np.ndarray) -> str:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if image_chw.shape[0] == 1:
            image_hwc = image_chw[0]
            pil = Image.fromarray(self._to_uint8(image_hwc), mode="L")
        else:
            image_hwc = np.transpose(image_chw, (1, 2, 0))
            pil = Image.fromarray(self._to_uint8(image_hwc), mode="RGB")

        out_path = run_dir / f"{name}.png"
        pil.save(out_path)
        return str(out_path)
