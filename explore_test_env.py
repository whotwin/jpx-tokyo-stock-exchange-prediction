"""
探索 JPX 赛事测试环境数据结构
"""
import pandas as pd
import os

# 尝试导入赛事环境
try:
    import jpx_tokyo_market_prediction
    env_available = True
except ImportError:
    env_available = False
    print("⚠️ 赛事环境不可用 (仅在 Kaggle 环境中可用)")

# 检查 example_test_files 目录
example_test_dir = "example_test_files"
if os.path.exists(example_test_dir):
    print("\n📁 example_test_files 目录内容:")
    for f in os.listdir(example_test_dir):
        filepath = os.path.join(example_test_dir, f)
        if f.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"\n  📄 {f}")
            print(f"     形状: {df.shape}")
            print(f"     列: {list(df.columns)}")
        else:
            print(f"  📄 {f}")
else:
    print("⚠️ example_test_files 目录不存在")

# 检查 supplemental_files 目录
supplemental_dir = "supplemental_files"
if os.path.exists(supplemental_dir):
    print("\n📁 supplemental_files 目录内容:")
    for f in os.listdir(supplemental_dir):
        filepath = os.path.join(supplemental_dir, f)
        if f.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"\n  📄 {f}")
            print(f"     形状: {df.shape}")
            print(f"     列: {list(df.columns)}")
            if 'Date' in df.columns:
                print(f"     日期范围: {df['Date'].min()} ~ {df['Date'].max()}")
else:
    print("⚠️ supplemental_files 目录不存在")

# 如果赛事环境可用，展示 iter_test() 返回的数据结构
if env_available:
    print("\n" + "="*60)
    print("🔍 赛事测试环境数据探索")
    print("="*60)

    try:
        env = jpx_tokyo_market_prediction.make_env()
        iter_test = env.iter_test()

        # 获取第一批测试数据
        first_batch = next(iter_test)

        print(f"\n📦 iter_test() 返回 {len(first_batch)} 个数据框:")
        for i, df in enumerate(first_batch):
            if hasattr(df, 'shape'):
                print(f"\n  [{i}] 类型: {type(df).__name__}")
                print(f"      形状: {df.shape}")
                print(f"      列: {list(df.columns)}")
                if 'Date' in df.columns:
                    print(f"      日期: {df['Date'].iloc[0] if len(df) > 0 else 'N/A'}")
                    print(f"      股票数: {len(df)}")
            else:
                print(f"  [{i}] {type(df).__name__}: {df}")

        # 保存样本数据
        print("\n💾 保存测试数据样本...")
        for i, df in enumerate(first_batch):
            if hasattr(df, 'shape') and df.shape[0] > 0:
                filename = f"test_sample_batch_{i}.csv"
                df.to_csv(filename, index=False)
                print(f"   已保存: {filename}")

    except Exception as e:
        print(f"❌ 探索失败: {e}")
else:
    print("\n📝 赛事环境说明:")
    print("   jpx_tokyo_market_prediction.make_env() 是 Kaggle 专用API")
    print("   在本地环境无法运行，仅用于提交预测结果")
    print("\n   提交格式要求:")
    print("   - Date: 交易日期")
    print("   - SecuritiesCode: 股票代码")
    print("   - Rank: 排名 (0=预期收益最高)")

print("\n" + "="*60)
