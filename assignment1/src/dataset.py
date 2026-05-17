from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    class_name: str


class OxfordPetClassificationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        images_dir: str,
        annotations_dir: str,
        transform: Callable | None = None,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.images_root = (self.root / images_dir).resolve()
        self.annotations_root = (self.root / annotations_dir).resolve()
        self.transform = transform
        self.split = split
        self.idx_to_class = self._load_all_classes()

        split_file = "trainval.txt" if split == "train" else "test.txt"
        self.samples = self._load_split(split_file)
        if limit is not None:
            self.samples = self.samples[:limit]

    def _load_split(self, split_file: str) -> list[Sample]:
        path = self.annotations_root / split_file
        samples: list[Sample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_stem, class_id, _, _ = line.split()
                class_index = int(class_id) - 1
                class_name = self._extract_class_name(image_stem)
                image_path = self.images_root / f"{image_stem}.jpg"
                if image_path.exists():
                    samples.append(
                        Sample(
                            image_path=image_path,
                            label=class_index,
                            class_name=self.idx_to_class[class_index],
                        )
                    )
        return samples

    def _load_all_classes(self) -> list[str]:
        list_path = self.annotations_root / "list.txt"
        mapping: dict[int, str] = {}
        with list_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                image_stem, class_id, _, _ = line.split()
                class_index = int(class_id) - 1
                mapping[class_index] = self._extract_class_name(image_stem)
        return [mapping[i] for i in sorted(mapping)]

    @staticmethod
    def _extract_class_name(image_stem: str) -> str:
        parts = image_stem.split("_")
        return "_".join(parts[:-1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "path": str(sample.image_path),
            "class_name": sample.class_name,
        }
