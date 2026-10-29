import pandas as pd
import streamlit as st
from pathlib import Path


DATA_DIR = Path(__file__).parent/'data'/'raw'


@st.cache_data
def load_data()->pd.DataFrame:
    #讀入資料表
    orders = pd.read_csv(DATA_DIR/"orders.csv", parse_dates=['order_date'])
    order_items = pd.read_csv(DATA_DIR/"order_items.csv")
    customers = pd.read_csv(DATA_DIR/"customers.csv")
    products = pd.read_csv(DATA_DIR/"products.csv")


    #合併資料:訂單明細+產品 -> +訂單 +顧客資料
    sales = (
        order_items.merge(
            products[['product_id','category']],
            on = 'product_id',
            how = 'left',
            validate = 'many_to_one'
            )
            .merge(
                orders, on="order_id",how='left',validate = 'many_to_one'
            )
            .merge(
                customers, on="customer_id",how='left',validate = 'many_to_one'
            )
    )
    #計算營收
    sales['revenue']=sales['quantity']*sales['unit_price']*(1-sales['discount_rate'])
    #返回整合後的資料表
    return sales


def format_money(value:float)->str:
    return f"NT${value:,.2f}" #將小數格式化成文字輸出


def main()->None: #主執行區
    st.title("營運狀況儀表板")
    st.caption("合併資料計算結果")


    sales = load_data()
    completed = sales.loc[sales['status'].eq('completed')].copy()
   
    #側邊欄
    st.sidebar.header("篩選條件")
    min_date = completed['order_date'].min().date()
    max_date = completed['order_date'].max().date()
    date_range = st.sidebar.date_input(#日期選擇器
        "選擇訂單日期",
        value = (min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    segments = sorted(completed['segment'].dropna().unique())#篩選客群名單
    categories = sorted(completed['category'].dropna().unique())#篩選商品類型
    selected_segments = st.sidebar.multiselect('客戶分群', segments,
                                               default=segments)
    selected_categories = st.sidebar.multiselect('商品類型', categories,
                                               default=categories)
    #判斷(排除非正確的資料)
    filtered = completed[completed['segment'].isin(selected_segments) & completed['category'].isin(selected_categories)]
   
    if len(date_range)==2:
        start_date, end_date = date_range
        filtered = filtered[filtered['order_date'].dt.date.between(start_date, end_date)]
   
    if filtered.empty:
        st.warning("沒有篩選資料...")
        st.stop() #中斷執行


    #各種計算
    #st.write(filtered.info())
    total_revenue = filtered['revenue'].sum()
    order_count = filtered['order_id'].nunique()
    customer_count = filtered['customer_id'].nunique()
    aver_order = total_revenue / order_count


    a,b,c,d = st.columns(4)
    a.metric("總營收",format_money(total_revenue))
    b.metric("訂單總數",f"{order_count:,}")
    c.metric("總客量",f"{customer_count:,}")
    d.metric("平均訂單金額",format_money(aver_order))


    left, right = st.columns(2)
    with left:
        st.subheader("營收趨勢")
        daily_revenue = (
            filtered.groupby('order_date',as_index=False)['revenue'].sum().set_index('order_date')
        )
        st.line_chart(daily_revenue, y='revenue', y_label="營收")


    with right:
        st.subheader("各類商品營收")
        category_revenue = (
            filtered.groupby('category',as_index=False)['revenue']
            .sum()
            .sort_values('revenue',ascending=False)
            .set_index('category')
        )
        st.bar_chart(category_revenue, y='revenue', y_label="營收")


    st.subheader("客群分析")
    segment_summary = (
        filtered.groupby('segment',as_index=False)
        .agg(
            訂單數 = ('order_id','nunique'),
            顧客數 = ('customer_id','nunique'),
            營收 = ('revenue','sum')
        ).sort_values('營收', ascending=False)
    )
    segment_summary['平均訂單金額']=segment_summary['營收']/segment_summary['訂單數']
    st.dataframe(
        segment_summary.style.format({'訂單數':'{:,}','營收':'{:,.0f}', '平均訂單金額':'{:,.0f}'    })
    ,hide_index=True
    )






if __name__=="__main__":
    main()





