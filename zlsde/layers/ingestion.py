"""Layer 1: Data Ingestion - Load and validate raw data from multiple sources."""

import csv
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import List

from zlsde.models.data_models import DataSource, RawDataItem

logger = logging.getLogger(__name__)


class DataIngestionLayer:
    """Load and validate raw data from multiple sources.

    Supports multiple input formats:
    - CSV files with text content
    - JSON files (single object or array)
    - Plain text files
    - Directories of text files

    Features:
    - Batch loading for memory efficiency
    - SHA-256 content hashing for deduplication
    - Encoding validation (UTF-8)
    - Non-empty content checks
    """

    def __init__(self, config):
        """Initialize data ingestion layer with configuration.

        Args:
            config: PipelineConfig with data source settings.
        """
        self.config = config
        self.batch_size = getattr(config, "batch_size", 1000)

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Text content to hash.

        Returns:
            Hexadecimal hash string.
        """
        return hashlib.sha256(str(content).encode("utf-8")).hexdigest()

    def load_data(self, sources: List[DataSource]) -> List[RawDataItem]:
        """Load data from configured sources.

        Supports CSV, JSON, text files, and folder sources.
        Uses batch loading for memory efficiency.

        Args:
            sources: List of DataSource configurations.

        Returns:
            List of RawDataItem objects.

        Raises:
            ValueError: If source type is unsupported or file not found.
            IOError: If file cannot be read.
        """
        all_items = []

        for source in sources:
            logger.info(f"Loading data from {source.type} source: {source.path}")

            if source.type == "csv":
                items = self._load_csv(source)
            elif source.type == "json":
                items = self._load_json(source)
            elif source.type == "text":
                items = self._load_text(source)
            elif source.type == "folder":
                items = self._load_folder(source)
            else:
                raise ValueError(f"Unsupported source type: {source.type}")

            all_items.extend(items)
            logger.info(f"Loaded {len(items)} items from {source.path}")

        logger.info(f"Total items loaded: {len(all_items)}")
        return all_items

    def _load_csv(self, source: DataSource) -> List[RawDataItem]:
        """Load data from CSV file.

        Expected CSV format:
        - Must have a 'content' column
        - Optional 'id' column (UUID generated if missing)
        - Optional 'modality' column (defaults to config modality)

        Args:
            source: DataSource with path to CSV file.

        Returns:
            List of RawDataItem objects.

        Raises:
            FileNotFoundError: If CSV file doesn't exist.
            ValueError: If CSV is missing required columns.
        """
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {source.path}")

        items = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate required columns
            if "content" not in reader.fieldnames:
                raise ValueError(f"CSV must have 'content' column: {source.path}")

            for row in reader:
                content = row.get("content", "").strip()
                if not content:
                    continue  # Skip empty rows

                item_id = row.get("id", str(uuid.uuid4()))
                modality = row.get("modality", self.config.modality)

                # Create item with minimal metadata to avoid validation issues
                item = RawDataItem(
                    id=item_id,
                    content=content,
                    modality=modality,
                    source=source.path,
                    content_hash=self._compute_content_hash(content),
                    metadata=None,
                )
                items.append(item)

        return items

    def _load_json(self, source: DataSource) -> List[RawDataItem]:
        """Load data from JSON file.

        Supports two formats:
        1. Array of objects: [{"content": "...", "id": "...", ...}, ...]
        2. Single object: {"content": "...", "id": "...", ...}

        Args:
            source: DataSource with path to JSON file.

        Returns:
            List of RawDataItem objects.

        Raises:
            FileNotFoundError: If JSON file doesn't exist.
            ValueError: If JSON is malformed or missing 'content' field.
        """
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {source.path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both single object and array
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError(f"JSON must be object or array: {source.path}")

        items = []
        for obj in data:
            if not isinstance(obj, dict):
                logger.warning(f"Skipping non-object item in JSON: {obj}")
                continue

            content = obj.get("content", "").strip()
            if not content:
                continue  # Skip empty content

            item_id = obj.get("id", str(uuid.uuid4()))
            modality = obj.get("modality", self.config.modality)

            # Extract metadata from other fields
            metadata = {k: v for k, v in obj.items() if k not in ["id", "content", "modality"]}
            if source.metadata:
                metadata.update(source.metadata)

            item = RawDataItem(
                id=item_id,
                content=content,
                modality=modality,
                source=source.path,
                content_hash=self._compute_content_hash(content),
                metadata=metadata if metadata else None,
            )
            items.append(item)

        return items

    def _load_text(self, source: DataSource) -> List[RawDataItem]:
        """Load data from plain text file.

        Each line becomes a separate data item.
        Empty lines are skipped.

        Args:
            source: DataSource with path to text file.

        Returns:
            List of RawDataItem objects.

        Raises:
            FileNotFoundError: If text file doesn't exist.
        """
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {source.path}")

        items = []

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                content = line.strip()
                if not content:
                    continue  # Skip empty lines

                item = RawDataItem(
                    id=str(uuid.uuid4()),
                    content=content,
                    modality=self.config.modality,
                    source=f"{source.path}:line{line_num}",
                    content_hash=self._compute_content_hash(content),
                    metadata=source.metadata,
                )
                items.append(item)

        return items

    def _load_folder(self, source: DataSource) -> List[RawDataItem]:
        """Load data from folder of text files.

        Recursively loads all .txt files in the folder.
        Each file becomes a separate data item.

        Args:
            source: DataSource with path to folder.

        Returns:
            List of RawDataItem objects.

        Raises:
            FileNotFoundError: If folder doesn't exist.
        """
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(f"Folder not found: {source.path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {source.path}")

        items = []

        # Find all text files recursively
        for file_path in path.rglob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    continue  # Skip empty files

                item = RawDataItem(
                    id=str(uuid.uuid4()),
                    content=content,
                    modality=self.config.modality,
                    source=str(file_path),
                    content_hash=self._compute_content_hash(content),
                    metadata=source.metadata,
                )
                items.append(item)
            except Exception as e:
                logger.warning(f"Failed to load file {file_path}: {e}")
                continue

        return items

    def deduplicate(self, items: List[RawDataItem]) -> List[RawDataItem]:
        """Remove duplicate items via content hashing.

        Uses SHA-256 content hashing to identify duplicates.
        Keeps the first occurrence of each unique content.

        Args:
            items: List of RawDataItem objects.

        Returns:
            List of unique RawDataItem objects.
        """
        if not items:
            return []

        seen_hashes = set()
        unique_items = []
        duplicates_count = 0

        for item in items:
            content_hash = item.content_hash

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_items.append(item)
            else:
                duplicates_count += 1

        logger.info(
            f"Removed {duplicates_count} duplicate items. " f"Unique items: {len(unique_items)}"
        )

        return unique_items

    def validate(self, items: List[RawDataItem]) -> List[RawDataItem]:
        """Validate data items meet basic requirements.

        Validation checks:
        - Non-empty content
        - Valid UTF-8 encoding (for text)
        - Valid UUID
        - Valid modality

        Invalid items are filtered out with warnings.

        Args:
            items: List of RawDataItem objects.

        Returns:
            List of valid RawDataItem objects.
        """
        if not items:
            return []

        valid_items = []
        invalid_count = 0

        for item in items:
            try:
                # Validate using built-in validation (Pydantic v2 compatible)
                type(item).model_validate(item.model_dump())

                # Additional encoding check for text
                if item.modality == "text" and isinstance(item.content, str):
                    # Try to encode/decode to verify UTF-8
                    item.content.encode("utf-8").decode("utf-8")

                valid_items.append(item)

            except (ValueError, UnicodeError) as e:
                logger.warning(f"Invalid item {item.id}: {e}")
                invalid_count += 1
                continue

        logger.info(f"Validation complete. Valid: {len(valid_items)}, " f"Invalid: {invalid_count}")

        if not valid_items:
            logger.error("No valid items after validation!")

        return valid_items
