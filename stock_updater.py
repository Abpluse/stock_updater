#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票数据稳定获取脚本 (单线程稳定版)
数据范围：2018-01-01 至今（最近一个牛熊周期）
"""

import os
import sys
import time
import argparse
import threading
from datetime import timedelta
import pandas as pd
import numpy as np
import baostock as bs

# ==================== 配置区 ====================
SAVE_DIR = "./data_bs_cycle"
LIST_FILE = "stock_list_bs.csv"
START_DATE = "2018-01-01"
MIN_DATA_DAYS = 20
RETRY_TIMES = 2
REQUEST_INTERVAL = 0.5
# ================================================

thread_local = threading.local()


def get_bs_connection():
    if not hasattr(thread_local, "bs_api") or thread_local.bs_api is None:
        try:
            lg = bs.login()
            if lg.error_code == '0':
                thread_local.bs_api = True
                return True
            else:
                thread_local.bs_api = None
                return False
        except:
            thread_local.bs_api = None
            return False
    return True


def reset_bs_connection():
    thread_local.bs_api = None


def fetch_stock_list_bs():
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ baostock 登录失败: {lg.error_msg}")
        return pd.DataFrame()
    print("📋 正在获取全市场股票列表...")
    rs = bs.query_all_stock(day="2024-06-26")
    if rs.error_code != '0':
        print(f"❌ 获取股票列表失败: {rs.error_msg}")
        bs.logout()
        return pd.DataFrame()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    if not data_list:
        return pd.DataFrame()
    df = pd.DataFrame(data_list, columns=rs.fields)
    df = df[df['code'].str.startswith(('sh.', 'sz.'))]
    df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
    df.rename(columns={'code_name': 'name'}, inplace=True)
    df = df[['code', 'name']]
    print(f"✅ 共获取 {len(df)} 只股票")
    return df


def fetch_kline_bs(code, start_date, end_date=None, retry=RETRY_TIMES):
    if end_date is None:
        end_date = time.strftime('%Y-%m-%d')
    for attempt in range(retry):
        try:
            if not get_bs_connection():
                reset_bs_connection()
                time.sleep(0.3 * (attempt + 1))
                continue
            bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields="date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"
            )
            if rs.error_code != '0':
                if attempt < retry - 1:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                return pd.DataFrame()
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            df = pd.DataFrame(data_list, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['open', 'high', 'low', 'close'], how='all')
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except Exception as e:
            reset_bs_connection()
            if attempt < retry - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            break
    return pd.DataFrame()


def calc_indicators(df):
    if df.empty:
        return df
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['MA120'] = close.rolling(120).mean()
    df['MA250'] = close.rolling(250).mean()

    df['EMA5'] = close.ewm(span=5, adjust=False).mean()
    df['EMA10'] = close.ewm(span=10, adjust=False).mean()
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()
    df['EMA60'] = close.ewm(span=60, adjust=False).mean()

    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI14'] = 100 - (100 / (1 + gain / loss))

    df['BOLL_MID'] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['BOLL_UP'] = df['BOLL_MID'] + 2 * std
    df['BOLL_LOW'] = df['BOLL_MID'] - 2 * std

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df['RSV'] = (close - low_14) / (high_14 - low_14) * 100
    df['K'] = df['RSV'].ewm(span=3, adjust=False).mean()
    df['D'] = df['K'].ewm(span=3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    tp = (high + low + close) / 3
    tp_ma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI20'] = (tp - tp_ma) / (0.015 * mad + 1e-10)

    df['WILLR'] = (high_14 - close) / (high_14 - low_14) * (-100)
    df['BIAS20'] = (close - df['MA20']) / df['MA20'] * 100

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    df['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['ATR14'] = df['TR'].rolling(14).mean()

    direction = np.sign(close.diff())
    df['OBV'] = (volume * direction).cumsum()

    cum_vol = volume.cumsum()
    df['VWAP'] = (close * volume).cumsum() / cum_vol.replace(0, np.nan)

    return df.fillna(0)


def save_stock_data(code, name, df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    code_str = str(code).zfill(6)
    filename = f"{save_dir}/{code_str}.csv"
    df['code'] = code_str
    df['name'] = name
    other_cols = [col for col in df.columns if col not in ['date', 'name', 'code']]
    cols = ['date', 'name'] + other_cols + ['code']
    df = df[cols]
    df.to_csv(filename, index=False, encoding='utf-8-sig')


def download_one_stock(code, name, save_dir, start_date):
    code_str = str(code).zfill(6)
    filename = f"{save_dir}/{code_str}.csv"
    if os.path.exists(filename):
        return code_str, "skip", 0
    df = fetch_kline_bs(code_str, start_date, retry=RETRY_TIMES)
    if df.empty or len(df) < MIN_DATA_DAYS:
        return code_str, "fail", 0
    df = calc_indicators(df)
    save_stock_data(code, name, df, save_dir)
    return code_str, "success", len(df)


def single_full_download(stock_list):
    os.makedirs(SAVE_DIR, exist_ok=True)
    total = len(stock_list)
    success = skipped = failed = 0
    start_time = time.time()
    print(f"\n📥 开始顺序全量下载 (2018年至今)...")
    print(f"总股票数: {total}")
    for idx, row in stock_list.iterrows():
        code, name = row['code'], row['name']
        print(f"[{idx + 1}/{total}] {code}({name}) ", end="")
        code_str, status, count = download_one_stock(code, name, SAVE_DIR, START_DATE)
        if status == "success":
            success += 1
            print(f"✅ 成功 ({count} 条)")
        elif status == "skip":
            skipped += 1
            print(f"⏭️ 已存在")
        else:
            failed += 1
            print(f"❌ 失败")
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  📊 进度: {idx + 1}/{total} ({((idx + 1) / total) * 100:.1f}%) | "
                  f"速度: {rate:.2f} 只/秒 | 耗时: {elapsed / 60:.1f} 分钟")
        time.sleep(REQUEST_INTERVAL)
    elapsed = time.time() - start_time
    print(f"\n✅ 全量下载完成！")
    print(f"  成功: {success}，跳过: {skipped}，失败: {failed}，总计: {total}")
    print(f"  总耗时: {elapsed / 60:.2f} 分钟")


def single_incremental_update(stock_list):
    today = time.strftime('%Y-%m-%d')
    os.makedirs(SAVE_DIR, exist_ok=True)
    total = len(stock_list)
    success = skipped = failed = 0
    start_time = time.time()
    print(f"\n🔄 开始顺序增量更新...")
    print(f"总股票数: {total}")
    for idx, row in stock_list.iterrows():
        code, name = row['code'], row['name']
        code_str = str(code).zfill(6)
        filename = f"{SAVE_DIR}/{code_str}.csv"
        print(f"[{idx + 1}/{total}] {code}({name}) ", end="")
        if not os.path.exists(filename):
            code_str2, status, count = download_one_stock(code, name, SAVE_DIR, START_DATE)
            if status == "success":
                success += 1
                print(f"✅ 全量成功 ({count} 条)")
            else:
                failed += 1
                print(f"❌ 全量失败")
            time.sleep(REQUEST_INTERVAL)
            continue
        try:
            existing = pd.read_csv(filename)
            existing['date'] = pd.to_datetime(existing['date'])
            last_date = existing['date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            if start_date >= today:
                skipped += 1
                print(f"✅ 已是最新")
                continue
            new_df = fetch_kline_bs(code_str, start_date, today)
            if new_df.empty:
                skipped += 1
                print(f"✅ 无新增数据")
                continue
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=['date'], keep='last')
            merged = merged.sort_values('date').reset_index(drop=True)
            merged = calc_indicators(merged)
            save_stock_data(code, name, merged, SAVE_DIR)
            success += 1
            print(f"✅ 更新成功 (新增 {len(new_df)} 条)")
        except Exception as e:
            failed += 1
            print(f"❌ 更新异常: {e}")
            reset_bs_connection()
        time.sleep(REQUEST_INTERVAL)
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  📊 进度: {idx + 1}/{total} ({((idx + 1) / total) * 100:.1f}%) | "
                  f"耗时: {elapsed / 60:.1f} 分钟")
    elapsed = time.time() - start_time
    print(f"\n✅ 增量更新完成！")
    print(f"  更新: {success}，已最新: {skipped}，失败: {failed}，总计: {total}")
    print(f"  总耗时: {elapsed / 60:.2f} 分钟")


def main():
    parser = argparse.ArgumentParser(description='股票数据稳定获取 (2018年至今)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', help='强制全量下载')
    group.add_argument('--update', action='store_true', help='强制增量更新')
    args = parser.parse_args()
    print("=" * 50)
    print("🚀 股票数据稳定获取脚本")
    print("   数据范围: 2018-01-01 至今 (最近一个牛熊周期)")
    print(f"   数据存储目录: {SAVE_DIR}")
    print("=" * 50)
    if not os.path.exists(LIST_FILE):
        print("\n📋 本地股票列表不存在，从服务器获取...")
        stock_list = fetch_stock_list_bs()
        if stock_list.empty:
            print("❌ 获取股票列表失败")
            sys.exit(1)
        stock_list.to_csv(LIST_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 股票列表已保存 ({len(stock_list)} 只)")
    else:
        stock_list = pd.read_csv(LIST_FILE)
        print(f"📋 加载股票列表 ({len(stock_list)} 只)")
    if args.full:
        print("🚀 执行强制全量下载...")
        single_full_download(stock_list)
    elif args.update:
        print("🔄 执行强制增量更新...")
        single_incremental_update(stock_list)
    else:
        if not os.path.exists(SAVE_DIR) or not any(f.endswith('.csv') for f in os.listdir(SAVE_DIR)):
            print("📦 本地无数据，执行首次全量下载...")
            single_full_download(stock_list)
        else:
            print("🔄 本地已有数据，执行增量更新...")
            single_incremental_update(stock_list)
    print("\n🎉 任务结束")


if __name__ == "__main__":
    main()