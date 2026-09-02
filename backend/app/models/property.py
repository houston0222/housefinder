from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Property(Base):
    __tablename__ = "properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    property_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    canonical_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    og_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    og_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    listing_type: Mapped[str | None] = mapped_column(String(50), nullable=True)

    price_text: Mapped[str | None] = mapped_column(String(100), nullable=True)
    price_amount: Mapped[int | None] = mapped_column(Integer, nullable=True)

    location: Mapped[str | None] = mapped_column(Text, nullable=True)
    bedrooms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    property_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    agent: Mapped[str | None] = mapped_column(Text, nullable=True)

    main_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    key_features: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    listing_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    room_measurements: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

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