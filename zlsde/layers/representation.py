"""Layer 2: Representation Engine - Transform raw data into semantic embeddings."""

import logging
from typing import List, Optional

import numpy as np

from zlsde.models.data_models import RawDataItem

logger = logging.getLogger(__name__)


class RepresentationEngine:
    """Transform raw data into dense semantic embeddings.

    Supports:
    - Text embeddings via SentenceTransformers
    - Batch processing for efficiency
    - L2 normalization
    - Optional UMAP dimensionality reduction
    - Deterministic seed control
    - CPU/GPU device selection
    """

    def __init__(
        self, modality: str, model_name: str, device: str = "cpu", seed: Optional[int] = None
    ):
        """Initialize embedding model for specified modality.

        Args:
            modality: Data modality ("text", "image", "multimodal").
            model_name: Name of the embedding model to use.
            device: Device for computation ("cpu", "cuda", "mps").
            seed: Random seed for reproducibility (optional).

        Raises:
            ValueError: If modality is unsupported or model cannot be loaded.
        """
        self.modality = modality
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.model = None
        self.embedding_dim = None

        # Set random seed for reproducibility
        if self.seed is not None:
            self._set_seed(self.seed)

        # Load the appropriate model
        self._load_model()

        logger.info(
            f"RepresentationEngine initialized: modality={modality}, "
            f"model={model_name}, device={device}, embedding_dim={self.embedding_dim}"
        )

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value.
        """
        import random

        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Make operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_model(self) -> None:
        """Load embedding model based on modality.

        Raises:
            ValueError: If modality is unsupported.
            ImportError: If required libraries are not installed.
        """
        if self.modality == "text":
            self._load_text_model()
        elif self.modality == "image":
            raise NotImplementedError("Image embeddings not yet implemented")
        elif self.modality == "multimodal":
            raise NotImplementedError("Multimodal embeddings not yet implemented")
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

    def _load_text_model(self) -> None:
        """Load SentenceTransformer model for text embeddings.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for text embeddings. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading SentenceTransformer model: {self.model_name}")

        # Load model with device specification
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

    def embed(self, items: List[RawDataItem], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for data items.

        Processes items in batches for efficiency and applies L2 normalization.

        Args:
            items: List of RawDataItem objects to embed.
            batch_size: Number of items to process per batch.

        Returns:
            2D numpy array of shape (len(items), embedding_dim) with L2-normalized embeddings.

        Raises:
            ValueError: If items is empty or contains invalid data.
            RuntimeError: If embedding generation fails.

        Preconditions:
        - items is non-empty list of RawDataItem
        - All items have same modality
        - batch_size >= 1
        - Embedding model is loaded and initialized

        Postconditions:
        - Returns 2D numpy array with shape (len(items), embedding_dim)
        - All embeddings are L2-normalized
        - Deterministic output for same input and seed
        - No NaN or Inf values in output
        """
        if not items:
            raise ValueError("Cannot embed empty list of items")

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Validate all items have correct modality
        for item in items:
            if item.modality != self.modality:
                raise ValueError(
                    f"Item modality {item.modality} does not match engine modality {self.modality}"
                )

        logger.info(f"Embedding {len(items)} items with batch_size={batch_size}")

        # Extract content from items
        if self.modality == "text":
            contents = [item.content for item in items]
            embeddings = self._embed_text_batch(contents, batch_size)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        # Validate output
        assert embeddings.shape[0] == len(items), "Embedding count mismatch"
        assert embeddings.shape[1] == self.embedding_dim, "Embedding dimension mismatch"
        assert not np.any(np.isnan(embeddings)), "NaN values in embeddings"
        assert not np.any(np.isinf(embeddings)), "Inf values in embeddings"

        logger.info(f"Generated embeddings with shape {embeddings.shape}")

        return embeddings

    def _embed_text_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings for text data in batches.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per batch.

        Returns:
            2D numpy array of L2-normalized embeddings.
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization
            )

            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        return embeddings

    def reduce_dimensions(
        self, embeddings: np.ndarray, n_components: int = 50, seed: Optional[int] = None
    ) -> np.ndarray:
        """Optional UMAP dimensionality reduction.

        Reduces high-dimensional embeddings to lower dimensions while preserving
        local and global structure.

        Args:
            embeddings: 2D numpy array of embeddings to reduce.
            n_components: Target number of dimensions.
            seed: Random seed for UMAP (uses engine seed if not provided).

        Returns:
            2D numpy array of reduced embeddings with shape (n_samples, n_components).

        Raises:
            ValueError: If embeddings is invalid or n_components is too large.
            ImportError: If umap-learn is not installed.

        Preconditions:
        - embeddings is 2D numpy array with shape (n_samples, n_features)
        - n_components >= 1 and n_components < n_features
        - n_samples >= n_components

        Postconditions:
        - Returns 2D numpy array with shape (n_samples, n_components)
        - Deterministic output for same input and seed
        - No NaN or Inf values in output
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D array, got shape {embeddings.shape}")

        n_samples, n_features = embeddings.shape

        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        if n_components >= n_features:
            raise ValueError(
                f"n_components ({n_components}) must be less than n_features ({n_features})"
            )

        if n_samples < n_components:
            raise ValueError(f"n_samples ({n_samples}) must be >= n_components ({n_components})")

        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for dimensionality reduction. "
                "Install with: pip install umap-learn"
            )

        # Use engine seed if not provided
        if seed is None:
            seed = self.seed if self.seed is not None else 42

        logger.info(f"Reducing dimensions from {n_features} to {n_components} using UMAP")

        # Create UMAP reducer with deterministic settings
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=seed,
            n_jobs=1,  # Single thread for reproducibility
            verbose=False,
        )

        # Fit and transform embeddings
        reduced_embeddings = reducer.fit_transform(embeddings)

        # Validate output
        assert reduced_embeddings.shape == (
            n_samples,
            n_components,
        ), "Shape mismatch after reduction"
        assert not np.any(np.isnan(reduced_embeddings)), "NaN values after reduction"
        assert not np.any(np.isinf(reduced_embeddings)), "Inf values after reduction"

        logger.info(f"Dimensionality reduction complete. New shape: {reduced_embeddings.shape}")

        return reduced_embeddings
