from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data" / "raw"

# 各資料表實際需要的欄位:明確挑選,避免 merge 後出現 _x/_y 重複欄位
ORDER_ITEM_COLS = ["order_id", "product_id", "quantity", "unit_price", "discount_rate"]
PRODUCT_COLS = ["product_id", "category"]
ORDER_COLS = ["order_id", "customer_id", "order_date", "status"]
CUSTOMER_COLS = ["customer_id", "segment"]

UNKNOWN = "未分類"
RANGE_OPTIONS = {"全部期間": None, "最近 30 天": 30, "最近 90 天": 90,
                 "最近 365 天": 365, "自訂": "custom"}
GRANULARITY = {"日": ("D", 7), "週": ("W", 4), "月": ("MS", 3)}  # (resample 規則, 移動平均期數)


def check_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 缺少欄位:{missing}")


@st.cache_data
def load_data() -> pd.DataFrame:
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(DATA_DIR / "order_items.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")

    check_columns(order_items, ORDER_ITEM_COLS, "order_items.csv")
    check_columns(products, PRODUCT_COLS, "products.csv")
    check_columns(orders, ORDER_COLS, "orders.csv")
    check_columns(customers, CUSTOMER_COLS, "customers.csv")

    sales = (
        order_items[ORDER_ITEM_COLS]
        .merge(products[PRODUCT_COLS], on="product_id", how="left", validate="many_to_one")
        .merge(orders[ORDER_COLS], on="order_id", how="left", validate="many_to_one")
        .merge(customers[CUSTOMER_COLS], on="customer_id", how="left", validate="many_to_one")
    )
    sales["revenue"] = sales["quantity"] * sales["unit_price"] * (1 - sales["discount_rate"])
    sales["order_day"] = sales["order_date"].dt.normalize()
    # 缺值不丟棄,歸為「未分類」,確保篩選後總營收與原始資料對得上
    sales["segment"] = sales["segment"].fillna(UNKNOWN)
    sales["category"] = sales["category"].fillna(UNKNOWN)
    return sales


def data_updated_at() -> str:
    ts = (DATA_DIR / "orders.csv").stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_money(value: float) -> str:
    return f"NT${value:,.0f}"


def kpis(df: pd.DataFrame) -> dict:
    revenue = df["revenue"].sum()
    orders = df["order_id"].nunique()
    return {
        "revenue": revenue,
        "orders": orders,
        "customers": df["customer_id"].nunique(),
        "avg": revenue / orders if orders else 0.0,
    }


def pct_delta(cur: float, prev: float | None) -> str | None:
    if prev is None or prev == 0:
        return None
    return f"{(cur - prev) / prev:+.1%}"


def pick_date_range(min_date, max_date):
    choice = st.sidebar.selectbox("期間", list(RANGE_OPTIONS), index=3)
    spec = RANGE_OPTIONS[choice]
    if spec == "custom":
        picked = st.sidebar.date_input(
            "選擇訂單日期", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )
        if not isinstance(picked, (tuple, list)) or len(picked) != 2:
            st.info("請選擇結束日期")
            st.stop()
        return picked
    if spec is None:
        return min_date, max_date
    return max(min_date, max_date - timedelta(days=spec - 1)), max_date


def trend_chart(filtered: pd.DataFrame, granularity: str) -> alt.Chart:
    rule, window = GRANULARITY[granularity]
    trend = (
        filtered.set_index("order_day")["revenue"]
        .resample(rule).sum()
        .rename("revenue").reset_index()
        .rename(columns={"order_day": "period"})
    )
    trend["ma"] = trend["revenue"].rolling(window, min_periods=1).mean()

    base = alt.Chart(trend).encode(
        x=alt.X("period:T", title=None, axis=alt.Axis(format="%Y-%m", labelOverlap=True))
    )
    raw = base.mark_line(opacity=0.45, color="#4c9be8").encode(
        y=alt.Y("revenue:Q", title=None, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("period:T", title="期間", format="%Y-%m-%d"),
                 alt.Tooltip("revenue:Q", title="營收", format=",.0f"),
                 alt.Tooltip("ma:Q", title="移動平均", format=",.0f")],
    )
    smooth = base.mark_line(strokeWidth=3, color="#ff8c00").encode(y="ma:Q")
    return (raw + smooth).properties(height=320)


def category_chart(filtered: pd.DataFrame) -> alt.Chart:
    cat = filtered.groupby("category", as_index=False)["revenue"].sum()
    return (
        alt.Chart(cat)
        .mark_bar(color="#4c9be8")
        .encode(
            x=alt.X("revenue:Q", title=None),
            y=alt.Y("category:N", title=None, sort="-x"),
            tooltip=[alt.Tooltip("category:N", title="商品類型"),
                     alt.Tooltip("revenue:Q", title="營收", format=",.0f")],
        )
        .properties(height=320)
    )


def main() -> None:
    st.set_page_config(page_title="營運狀況儀表板", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {font-size: 1.9rem;}
        [data-testid="stMetricValue"] > div {
            overflow: visible; text-overflow: clip; white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("營運狀況儀表板")

    try:
        sales = load_data()
    except pd.errors.MergeError as e:
        st.error(f"資料合併失敗,請檢查是否有重複的主鍵:{e}")
        st.stop()
    except (KeyError, FileNotFoundError) as e:
        st.error(f"資料讀取失敗:{e}")
        st.stop()

    completed = sales.loc[sales["status"].eq("completed")].copy()
    if completed.empty:
        st.warning("沒有已完成的訂單。")
        st.stop()

    # ---- 側邊欄 ----
    st.sidebar.header("篩選條件")
    min_date = completed["order_day"].min().date()
    max_date = completed["order_day"].max().date()
    start_date, end_date = pick_date_range(min_date, max_date)

    segments = sorted(completed["segment"].unique())
    categories = sorted(completed["category"].unique())
    selected_segments = st.sidebar.multiselect("客戶分群", segments, default=segments)
    selected_categories = st.sidebar.multiselect("商品類型", categories, default=categories)
    granularity = st.sidebar.selectbox("趨勢圖粒度", list(GRANULARITY), index=1)

    # ---- 篩選(先篩客群/類別,再依日期切出本期與前期)----
    scoped = completed[
        completed["segment"].isin(selected_segments)
        & completed["category"].isin(selected_categories)
    ]

    def in_range(df: pd.DataFrame, s, e) -> pd.DataFrame:
        return df[(df["order_day"] >= pd.Timestamp(s)) & (df["order_day"] <= pd.Timestamp(e))]

    filtered = in_range(scoped, start_date, end_date)
    if filtered.empty:
        st.warning("沒有符合篩選條件的資料...")
        st.stop()

    # 前一期間:與本期等長、緊接在本期之前;若超出資料範圍則不比較
    days = (end_date - start_date).days + 1
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    prev = kpis(in_range(scoped, prev_start, prev_end)) if prev_start >= min_date else None

    st.caption(
        f"資料期間 {start_date} ~ {end_date}|僅計入狀態為 completed 的訂單|"
        f"營收 = 數量 × 單價 × (1 − 折扣率),未含運費、稅與退款|資料更新:{data_updated_at()}"
    )

    # ---- KPI ----
    cur = kpis(filtered)
    a, b, c, d = st.columns(4)
    a.metric("總營收", format_money(cur["revenue"]),
             pct_delta(cur["revenue"], prev and prev["revenue"]))
    b.metric("訂單總數", f"{cur['orders']:,}",
             pct_delta(cur["orders"], prev and prev["orders"]))
    c.metric("購買顧客數", f"{cur['customers']:,}",
             pct_delta(cur["customers"], prev and prev["customers"]),
             help="在目前篩選條件下,有下單的不重複顧客數,不是全體顧客數。")
    d.metric("平均訂單金額", format_money(cur["avg"]),
             pct_delta(cur["avg"], prev and prev["avg"]))
    if prev is None:
        st.caption("前一期間超出資料範圍,故不顯示變化率。")
    else:
        st.caption(f"變化率為與前一期間({prev_start} ~ {prev_end})比較。")

    # ---- 圖表 ----
    left, right = st.columns(2)
    with left:
        st.subheader("營收趨勢")
        st.caption(f"營收(NT$)|淺藍:每{granularity}營收;橘線:{GRANULARITY[granularity][1]} 期移動平均")
        st.altair_chart(trend_chart(filtered, granularity), use_container_width=True)
    with right:
        st.subheader("各類商品營收")
        st.caption("營收(NT$),由高到低排序")
        st.altair_chart(category_chart(filtered), use_container_width=True)

    # ---- 客群分析 ----
    st.subheader("客群分析")
    summary = (
        filtered.groupby("segment", as_index=False)
        .agg(訂單數=("order_id", "nunique"),
             顧客數=("customer_id", "nunique"),
             營收=("revenue", "sum"))
        .sort_values("營收", ascending=False)
        .rename(columns={"segment": "客群"})
    )
    summary[["訂單數", "顧客數"]] = summary[["訂單數", "顧客數"]].astype(int)
    summary["營收占比"] = summary["營收"] / summary["營收"].sum()
    summary["顧客占比"] = summary["顧客數"] / summary["顧客數"].sum()
    summary["每客營收"] = summary["營收"] / summary["顧客數"]
    summary["平均訂單金額"] = summary["營收"] / summary["訂單數"]

    top = summary.iloc[0]
    if len(summary) > 1 and top["營收占比"] > top["顧客占比"]:
        st.info(
            f"「{top['客群']}」客群占顧客 {top['顧客占比']:.0%},卻貢獻 {top['營收占比']:.0%} 的營收"
            f"(每位顧客平均 {format_money(top['每客營收'])})。"
        )

    st.dataframe(
        summary.style.format({
            "訂單數": "{:,}", "顧客數": "{:,}", "營收": "{:,.0f}",
            "營收占比": "{:.1%}", "顧客占比": "{:.1%}",
            "每客營收": "{:,.0f}", "平均訂單金額": "{:,.0f}",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "下載客群分析 (CSV)",
        summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="segment_summary.csv",
        mime="text/csv",
    )

    # ---- 資料對帳 ----
    with st.expander("資料對帳與品質檢查"):
        not_completed = sales[~sales["status"].eq("completed")]
        st.write(
            f"- 全部明細 {len(sales):,} 筆,其中 completed {len(completed):,} 筆;"
            f"未計入(非 completed){len(not_completed):,} 筆,"
            f"對應營收 {format_money(not_completed['revenue'].sum())}。"
        )
        unknown = completed[(completed["segment"] == UNKNOWN) | (completed["category"] == UNKNOWN)]
        if unknown.empty:
            st.write("- 已完成訂單中沒有客群或商品類型缺值。")
        else:
            st.warning(
                f"有 {len(unknown):,} 筆已完成明細的客群或商品類型缺值"
                f"(營收 {format_money(unknown['revenue'].sum())}),已歸類為「{UNKNOWN}」。"
            )
        st.write(
            f"- 已完成訂單總營收 {format_money(completed['revenue'].sum())};"
            f"目前篩選顯示 {format_money(cur['revenue'])}。"
        )
        if not not_completed.empty:
            st.dataframe(
                not_completed.groupby("status")["revenue"]
                .agg(筆數="size", 營收="sum").reset_index().rename(columns={"status": "狀態"}),
                hide_index=True,
            )


if __name__ == "__main__":
    main()
