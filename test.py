import requests
import packaging.version

def get_package_deps(package_name):
    print(f"\n{'='*20} 正在查询: {package_name} {'='*20}")
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"无法获取 {package_name} 的信息")
        return

    data = response.json()
    # 按版本号排序（从新到旧）
    versions = sorted(data["releases"].keys(), key=packaging.version.parse, reverse=True)

    target_deps = ['transformers', 'tokenizers', 'huggingface-hub', 'torch', 'python', '']

    print(f"{'版本号':<15} | {'核心依赖约束项'}")
    print("-" * 60)

    for v in versions:
        # 获取特定版本的详细 JSON
        v_url = f"https://pypi.org/pypi/{package_name}/{v}/json"
        v_data = requests.get(v_url).json()
        requires = v_data["info"].get("requires_dist")

        if not requires:
            print(f"{v:<15} | (无依赖信息或仅在 setup.py 中)")
            continue

        # 筛选我们关心的库
        relevant = [r for r in requires if any(dep in r.lower() for dep in target_deps)]
        dep_str = " ; ".join(relevant) if relevant else "未明确限制这三个库"
        print(f"{v:<15} | {dep_str}")

if __name__ == "__main__":
    # 需要先安装: pip install requests packaging
    get_package_deps("momentfm")
    get_package_deps("chronos-forecasting")