import pandas as pd

# 这是你截图里的真实 API 端点
base_url = "https://data.cityofnewyork.us/resource/2yzn-sicd.csv"

# 我们需要的数据总量
total_rows_needed = 100000
# 每次拉取 50000 行（分批拉取，防止网络超时报错）
chunk_size = 50000 

df_list =[]

print("🚀 开始从纽约市政府服务器拉取数据...")

# 循环拉取数据
for offset in range(0, total_rows_needed, chunk_size):
    print(f"正在拉取第 {offset} 行 到 第 {offset + chunk_size} 行数据...")
    
    # 核心魔法：用 $limit 突破 1000 行限制，用 $offset 翻页
    api_url = f"{base_url}?$limit={chunk_size}&$offset={offset}"
    
    # 读取这一小批数据
    chunk = pd.read_csv(api_url)
    df_list.append(chunk)
    
    # 如果拉回来的数据不够 50000 行，说明到底了，提前结束
    if len(chunk) < chunk_size:
        print("数据已全部拉取完毕！")
        break

# 把分批拉取的数据拼接到一起
final_df = pd.concat(df_list, ignore_index=True)

preview_columns = [
    col for col in ["pickup_datetime", "pickup_latitude", "pickup_longitude"]
    if col in final_df.columns
]

print(f"\n✅ 成功获取了 {len(final_df)} 行数据！")
print("数据前 5 行预览：")
print(final_df[preview_columns].head())

# 保存到本地电脑
final_df.to_csv("../data/NYC_YellowTaxi_2015_100k.csv", index=False)
print("\n💾 文件已保存为: NYC_YellowTaxi_2015_100k.csv，你可以开始切六边形网格了！")
