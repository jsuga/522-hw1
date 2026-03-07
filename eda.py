from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="price", nbins=80, title="Price Distribution")
    fig.update_layout(xaxis_title="Price (USD)", yaxis_title="Listings")
    return fig


def log_price_distribution(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["log_price"] = np.log1p(data["price"])
    fig = px.histogram(data, x="log_price", nbins=80, title="Log Price Distribution")
    fig.update_layout(xaxis_title="log(1 + price)", yaxis_title="Listings")
    return fig


def price_by_borough(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="neighbourhood_group",
        y="price",
        color="neighbourhood_group",
        points=False,
        title="Price by Borough",
    )
    fig.update_layout(showlegend=False)
    return fig


def price_by_room_type(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="room_type",
        y="price",
        color="room_type",
        points="outliers",
        title="Price by Room Type",
    )
    fig.update_layout(showlegend=False)
    return fig


def listings_by_borough(df: pd.DataFrame) -> go.Figure:
    counts = df["neighbourhood_group"].value_counts().rename_axis("borough").reset_index(name="count")
    fig = px.bar(counts, x="borough", y="count", color="borough", title="Listings by Borough")
    fig.update_layout(showlegend=False)
    return fig


def top_neighbourhoods_by_listing_count(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    counts = df["neighbourhood"].value_counts().head(top_n).rename_axis("neighbourhood").reset_index(name="count")
    fig = px.bar(
        counts.sort_values("count"),
        x="count",
        y="neighbourhood",
        orientation="h",
        title=f"Top {top_n} Neighbourhoods by Listing Count",
    )
    return fig


def median_price_by_borough_room(df: pd.DataFrame) -> go.Figure:
    table = (
        df.groupby(["neighbourhood_group", "room_type"], as_index=False)["price"]
        .median()
        .rename(columns={"price": "median_price"})
    )
    fig = px.density_heatmap(
        table,
        x="neighbourhood_group",
        y="room_type",
        z="median_price",
        color_continuous_scale="Viridis",
        title="Median Price by Borough and Room Type",
    )
    return fig


def minimum_nights_vs_price(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="minimum_nights",
        y="price",
        color="room_type",
        opacity=0.5,
        title="Minimum Nights vs Price",
    )
    fig.update_xaxes(type="log")
    return fig


def reviews_vs_price(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="reviews_per_month",
        y="price",
        color="neighbourhood_group",
        opacity=0.55,
        title="Reviews per Month vs Price",
    )
    return fig


def availability_vs_price(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="availability_365",
        y="price",
        color="neighbourhood_group",
        opacity=0.55,
        title="Availability vs Price",
    )
    return fig


def map_scatter(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(n=min(4000, len(df)), random_state=42)
    fig = px.scatter_mapbox(
        sample,
        lat="latitude",
        lon="longitude",
        color="price",
        size="price",
        size_max=12,
        zoom=9,
        height=640,
        title="Spatial Distribution of Listings (Price-Colored)",
        color_continuous_scale="Turbo",
    )
    fig.update_layout(mapbox_style="carto-positron")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include=["number"])
    corr = numeric.corr(numeric_only=True)
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
    fig.update_layout(height=620)
    return fig


def review_activity_over_time(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["last_review"] = pd.to_datetime(data["last_review"], errors="coerce")
    trend = (
        data.dropna(subset=["last_review"])
        .groupby(pd.Grouper(key="last_review", freq="M"))
        .size()
        .rename("review_count")
        .reset_index()
    )
    fig = px.line(trend, x="last_review", y="review_count", title="Review Activity Over Time")
    return fig
