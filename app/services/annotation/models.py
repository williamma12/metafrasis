"""
Annotation Data Models

Dataclasses for representing annotation datasets, images, and regions.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
import uuid
import json


@dataclass
class Point:
    """2D point with x, y coordinates"""
    x: float
    y: float

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Point":
        return cls(x=data["x"], y=data["y"])


@dataclass
class Region:
    """
    Annotated text region

    Attributes:
        id: Unique identifier for the region
        type: Shape type ("rectangle" or "polygon")
        points: List of points defining the region
        text: Optional text transcription
        auto_detected: Whether region was auto-detected by OCR
        verified: Whether region has been manually verified
        created_at: Timestamp when region was created
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal["rectangle", "polygon"] = "rectangle"
    points: List[Point] = field(default_factory=list)
    text: Optional[str] = None
    auto_detected: bool = False
    verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "points": [p.to_dict() for p in self.points],
            "text": self.text,
            "auto_detected": self.auto_detected,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Region":
        return cls(
            id=data["id"],
            type=data["type"],
            points=[Point.from_dict(p) for p in data["points"]],
            text=data.get("text"),
            auto_detected=data.get("auto_detected", False),
            verified=data.get("verified", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )

    @classmethod
    def from_bbox(cls, x: float, y: float, width: float, height: float, **kwargs) -> "Region":
        """Create a rectangle region from bounding box coordinates"""
        points = [
            Point(x, y),
            Point(x + width, y),
            Point(x + width, y + height),
            Point(x, y + height),
        ]
        return cls(type="rectangle", points=points, **kwargs)

    def get_bbox(self) -> tuple:
        """Get bounding box (x, y, width, height) from region points"""
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (x_min, y_min, x_max - x_min, y_max - y_min)


@dataclass
class AnnotatedImage:
    """
    An image with its annotations

    Attributes:
        id: Unique identifier for the image
        image_path: Path to the image file (relative to data directory)
        width: Image width in pixels
        height: Image height in pixels
        regions: List of annotated regions
    """
    id: str = field(default_factory=lambda: f"img_{uuid.uuid4().hex[:6]}")
    image_path: str = ""
    width: int = 0
    height: int = 0
    regions: List[Region] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotatedImage":
        return cls(
            id=data["id"],
            image_path=data["image_path"],
            width=data.get("width", 0),
            height=data.get("height", 0),
            regions=[Region.from_dict(r) for r in data.get("regions", [])],
        )

    def add_region(self, region: Region) -> None:
        """Add a region to this image"""
        self.regions.append(region)

    def remove_region(self, region_id: str) -> bool:
        """Remove a region by ID. Returns True if found and removed."""
        for i, region in enumerate(self.regions):
            if region.id == region_id:
                self.regions.pop(i)
                return True
        return False

    def get_region(self, region_id: str) -> Optional[Region]:
        """Get a region by ID"""
        for region in self.regions:
            if region.id == region_id:
                return region
        return None


@dataclass
class AnnotationDataset:
    """
    A dataset of annotated images

    Attributes:
        name: Dataset name (used as filename)
        version: Dataset version string
        created_at: When the dataset was created
        updated_at: When the dataset was last modified
        images: List of annotated images
    """
    name: str = "untitled"
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    images: List[AnnotatedImage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "images": [img.to_dict() for img in self.images],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationDataset":
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            images=[AnnotatedImage.from_dict(img) for img in data.get("images", [])],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AnnotationDataset":
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def add_image(self, image: AnnotatedImage) -> None:
        """Add an image to the dataset"""
        self.images.append(image)
        self.updated_at = datetime.now()

    def remove_image(self, image_id: str) -> bool:
        """Remove an image by ID. Returns True if found and removed."""
        for i, image in enumerate(self.images):
            if image.id == image_id:
                self.images.pop(i)
                self.updated_at = datetime.now()
                return True
        return False

    def get_image(self, image_id: str) -> Optional[AnnotatedImage]:
        """Get an image by ID"""
        for image in self.images:
            if image.id == image_id:
                return image
        return None

    def get_image_by_path(self, image_path: str) -> Optional[AnnotatedImage]:
        """Get an image by its path"""
        for image in self.images:
            if image.image_path == image_path:
                return image
        return None

    @property
    def total_regions(self) -> int:
        """Total number of regions across all images"""
        return sum(len(img.regions) for img in self.images)

    @property
    def labeled_regions(self) -> int:
        """Number of regions with text labels"""
        return sum(
            1 for img in self.images
            for region in img.regions
            if region.text
        )

    @property
    def verified_regions(self) -> int:
        """Number of verified regions"""
        return sum(
            1 for img in self.images
            for region in img.regions
            if region.verified
        )
