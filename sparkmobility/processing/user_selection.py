import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

sns.set(style="whitegrid", font_scale=1.5)
custom_colors = [
    "#c45161",
    "#e094a0",
    "#f2b6c0",
    "#f2dde1",
    "#cbc7d8",
    "#8db7d2",
    "#5e62a9",
    "#434279",
]
cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
matplotlib.rcParams.update({"legend.fontsize": 14, "legend.handlelength": 2})


class UserSelection:
    def __init__(
        self,
        MobilityDataset,
    ):
        self.dataset = MobilityDataset

    def filter_users(
        self,
        num_stay_points_range,
        time_span_days_range,
    ):
        df = self.dataset.load_stays()

        df_grouped = (
            df.groupBy("caid")
            .agg(
                F.count("*").alias("num_stays"),
                F.min("stay_start_timestamp").alias("first_stay"),
                F.max("stay_end_timestamp").alias("last_stay"),
            )
            .withColumn(
                "duration_days",
                F.round(
                    (F.unix_timestamp("last_stay") - F.unix_timestamp("first_stay"))
                    / 86400
                ).cast("int"),
            )
        ).cache()

        active_level = (
            df_grouped.groupBy("duration_days", "num_stays")
            .count()
            .orderBy("duration_days", "num_stays")
            .toPandas()
        )

        filter_condition = (
            (F.col("num_stays") >= num_stay_points_range[0])
            & (F.col("num_stays") <= num_stay_points_range[1])
            & (F.col("duration_days") >= time_span_days_range[0])
            & (F.col("duration_days") <= time_span_days_range[1])
        )

        selected_users = df_grouped.filter(filter_condition).select("caid")

        filtered_df = df.join(selected_users, on="caid", how="inner")

        # df_grouped has one row per caid, so its row count equals the distinct
        # user count. Computing both totals in a single agg avoids the extra
        # distinct+count and filter+count shuffles.
        counts = df_grouped.agg(
            F.count("*").alias("total"),
            F.sum(F.when(filter_condition, 1).otherwise(0)).alias("filtered"),
        ).first()
        self.dataset.num_total_users = counts["total"]
        self.dataset.num_filtered_users = counts["filtered"]

        filtered_df.write.mode("overwrite").parquet(
            self.dataset.output_path + "/FilteredUserStayPoints"
        )

        fig, ax = self._visualize(
            active_level, num_stay_points_range, time_span_days_range
        )

        out_path = self.dataset.output_path + "/FilteredUserStayPoints"
        print(f"Total users: {self.dataset.num_total_users}")
        print(f"Filtered users: {self.dataset.num_filtered_users}")
        print(f" Saved to: {out_path}")
        print(f"Filtered users saved to: {out_path}")
        df_grouped.unpersist()
        return fig, ax

    @staticmethod
    def _visualize(active_level, num_stay_points_range, time_span_days_range):
        # One pivot cell per exact stay count and day is unreadable once users
        # spread over hundreds of values; bin both axes (10 stays, 5 days) when
        # they do. Tick labels and the selection box follow the binned values.
        active_level = active_level.copy()
        if active_level["num_stays"].nunique() > 100:
            # a handful of extreme users would otherwise stretch the x-axis and
            # squeeze everyone else into its left edge; the last column collects them
            cap = int(
                np.repeat(active_level["num_stays"], active_level["count"]).quantile(0.99)
            )
            active_level["num_stays"] = active_level["num_stays"].clip(upper=cap)
            active_level["num_stays"] = (active_level["num_stays"] // 10) * 10
        if active_level["duration_days"].nunique() > 60:
            active_level["duration_days"] = (active_level["duration_days"] // 5) * 5
        active_level = active_level.groupby(
            ["duration_days", "num_stays"], as_index=False
        )["count"].sum()
        pivot = active_level.pivot(
            index="duration_days", columns="num_stays", values="count"
        ).fillna(0)
        log_data = np.log10(pivot + 1)
        mask = pivot.values == 0
        masked_data = np.ma.masked_where(mask, log_data.values)

        cmap_masked = LinearSegmentedColormap.from_list(
            "custom_cmap_masked", custom_colors
        )
        cmap_masked.set_bad("white")

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.grid(False)
        ax.minorticks_off()
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        im = ax.imshow(masked_data, aspect="auto", cmap=cmap_masked)

        # imshow places cell k of the pivot at coordinate k, so the selection box
        # must be drawn at the positions of the requested values among the
        # distinct values present, not at the values themselves.
        cols = log_data.columns.values
        rows = log_data.index.values
        x0 = np.searchsorted(cols, num_stay_points_range[0]) - 0.5
        x1 = np.searchsorted(cols, num_stay_points_range[1], side="right") - 0.5
        y0 = np.searchsorted(rows, time_span_days_range[0]) - 0.5
        y1 = np.searchsorted(rows, time_span_days_range[1], side="right") - 0.5
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor=custom_colors[-1],
            facecolor="none",
            clip_on=False,
            zorder=5,  # above the axes frame, so edges on the plot boundary stay visible
        )
        ax.add_patch(rect)

        num_cols = log_data.shape[1]
        if num_cols > 10:
            xticks = np.linspace(0, num_cols - 1, 10, dtype=int)
        else:
            xticks = np.arange(num_cols)
        ax.set_xticks(xticks)
        ax.set_xticklabels(log_data.columns[xticks], rotation=90)

        num_rows = log_data.shape[0]
        if num_rows > 10:
            yticks = np.linspace(0, num_rows - 1, 10, dtype=int)
        else:
            yticks = np.arange(num_rows)

        ax.set_yticks(yticks)
        ax.set_yticklabels(log_data.index[yticks])
        ax.invert_yaxis()

        ax.set_xlabel("Number of Stay Points")
        ax.set_ylabel("Time Span (Days)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Log10 Number of Users")
        plt.tight_layout()

        return fig, ax
