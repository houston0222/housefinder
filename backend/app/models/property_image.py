from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.property import Base

class PropertyImage(Base):
    __tablename__ = "property_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    property_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("properties.property_id", ondelete="CASCADE"),
        index=True,
    )

    image_id: Mapped[str] = mapped_column(String(50), index=True)

    source_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    image_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    room_or_area: Mapped[str | None] = mapped_column(String(255), nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)

    search_phrases: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    visual_observations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    capacity_estimates: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

    overall_confidence: Mapped[float | None] = mapped_column(nullable=True)

    embedding_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("property_id", "image_id", name="uq_property_images_property_id_image_id"),
    )