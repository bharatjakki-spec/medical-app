"""
pipeline.py
-----------
End-to-end Medical Image Co-Segmentation Pipeline.

Usage
-----
    from pipeline import MedicalCoSegPipeline

    pipeline = MedicalCoSegPipeline(n_clusters=5)
    results  = pipeline.run_on_paths(["scan1.png", "scan2.png"])
    # OR
    results  = pipeline.run_on_arrays([bgr_array1, bgr_array2])
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm

from medical_coseg.preprocessing   import MedicalImagePreprocessor
from medical_coseg.feature_extractor import ResNetFeatureExtractor
from medical_coseg.clustering        import DeepKMeans
from medical_coseg.postprocessing    import SegmentationPostProcessor
from medical_coseg.visualization     import SegmentationVisualizer


class MedicalCoSegPipeline:
    """
    Full co-segmentation pipeline: load -> preprocess -> extract -> cluster
    -> postprocess -> visualize.

    Parameters
    ----------
    n_clusters : int
        Number of semantic regions to discover (e.g. 5 for tissue types).
    target_size : tuple(int, int)
        Image resize target for preprocessing, e.g. (224, 224).
    feature_size : tuple(int, int)
        Spatial resolution of extracted feature maps, e.g. (56, 56).
    use_pca : bool
        Apply PCA before clustering to reduce memory and speed up fitting.
    pca_components : int
        PCA output dimensions (only if use_pca=True).
    use_minibatch : bool
        Use MiniBatchKMeans for speed on large datasets.
    overlay_alpha : float
        Colour overlay blend strength [0, 1].
    output_dir : str
        Directory for saving output visualisations.
    use_gpu : bool
        Use CUDA GPU if available.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        target_size: tuple = (224, 224),
        feature_size: tuple = (56, 56),
        use_pca: bool = True,
        pca_components: int = 64,
        use_minibatch: bool = True,
        overlay_alpha: float = 0.55,
        output_dir: str = "outputs",
        use_gpu: bool = True,
    ):
        self.n_clusters  = n_clusters
        self.feature_size = feature_size
        self.output_dir  = output_dir

        print("\n" + "="*60)
        print("  Medical Image Co-Segmentation Pipeline")
        print("  Core: ResNet-50 Features + K-Means Clustering")
        print("="*60)

        print("\n[1/5] Initialising preprocessor ...")
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        print("[2/5] Loading ResNet-50 feature extractor ...")
        self.extractor = ResNetFeatureExtractor(
            output_size=feature_size,
            use_gpu=use_gpu,
        )

        print("[3/5] Configuring K-Means clusterer ...")
        self.clusterer = DeepKMeans(
            n_clusters=n_clusters,
            use_pca=use_pca,
            pca_components=pca_components,
            use_minibatch=use_minibatch,
        )

        self.postprocessor = SegmentationPostProcessor(overlay_alpha=overlay_alpha)
        self.visualizer    = SegmentationVisualizer(output_dir=output_dir)

        print(f"[Pipeline ready] k={n_clusters}, feature_size={feature_size}\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_on_paths(self, image_paths: List[Union[str, Path]]) -> dict:
        """
        Run the full pipeline on a list of image file paths.

        Returns
        -------
        dict with keys: originals, overlays, color_maps, label_maps, output_paths
        """
        print(f"[4/5] Loading & preprocessing {len(image_paths)} images ...")
        originals, tensors = self.preprocessor.batch_load(image_paths)
        return self._run_core(originals, tensors)

    def run_on_arrays(self, bgr_arrays: List[np.ndarray]) -> dict:
        """
        Run the full pipeline on a list of BGR numpy arrays (uint8).
        Useful for synthetic demos or in-memory image data.
        """
        print(f"[4/5] Preprocessing {len(bgr_arrays)} in-memory images ...")
        originals, tensors_list = [], []
        for arr in bgr_arrays:
            orig, t = self.preprocessor.preprocess_array(arr)
            originals.append(orig)
            tensors_list.append(t)

        import torch
        tensors = torch.cat(tensors_list, dim=0)
        return self._run_core(originals, tensors)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_core(self, originals, tensors) -> dict:
        """Shared core: extract features, cluster, postprocess, visualise."""

        print("[4/5] Extracting multi-scale ResNet-50 features ...")
        features = self.extractor.extract(tensors)   # (N, H*W, C)
        print(f"      Feature array shape: {features.shape}")

        print("[5/5] Running K-Means co-segmentation ...")
        labels     = self.clusterer.fit_predict(features)     # (N, H*W)
        label_maps = self.clusterer.labels_to_maps(           # (N, H, W)
            labels, self.feature_size
        )

        print("\n[Post] Applying morphological clean-up & colour coding ...")
        batch_results = self.postprocessor.process_batch(originals, label_maps)
        overlays   = [r[0] for r in batch_results]
        color_maps = [r[1] for r in batch_results]

        print("[Post] Saving visualisations ...")
        output_paths = []

        for i, (orig, cmap, overlay) in enumerate(zip(originals, color_maps, overlays)):
            p = self.visualizer.save_single(
                orig, cmap, overlay,
                filename=f"scan_{i+1:03d}_segmentation.png",
                title=f"Scan {i+1} - Co-Segmentation (k={self.n_clusters})",
            )
            output_paths.append(p)

        # Co-segmentation grid (all scans together)
        grid_path = self.visualizer.save_coseg_grid(
            originals, overlays, self.n_clusters
        )
        output_paths.append(grid_path)

        # Cluster statistics bar chart
        stats_path = self.visualizer.save_cluster_stats(label_maps, self.n_clusters)
        output_paths.append(stats_path)

        print("\n" + "="*60)
        print(f"  DONE! Results saved to: {self.output_dir}/")
        print("="*60 + "\n")

        return {
            "originals":   originals,
            "overlays":    overlays,
            "color_maps":  color_maps,
            "label_maps":  label_maps,
            "output_paths": output_paths,
        }
