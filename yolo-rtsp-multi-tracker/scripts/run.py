"""入口引导文件。

用法:
    python run.py                        # 使用 settings.json 默认配置
    python run.py --mode tracking        # 多摄追踪模式
    python run.py --mode spatial --show  # 单摄空间检测 + 预览

功能
----
将 scripts/ 目录加入 sys.path，使子包（core/detection/tracking/trajectory/agent）
的绝对导入能正确解析，然后委托给 main.py。
"""
import os
import sys

# 将 scripts/ 目录加入模块搜索路径，确保子包 import 可用
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
