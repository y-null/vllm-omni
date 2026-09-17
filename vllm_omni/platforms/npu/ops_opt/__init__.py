"""算子项源码落盘目录（Triton 内核与 C++ 扩展）。

定位：只落源码、全部默认 OFF（各自模块级 env 门）、未与运行时接线。
将来集成点在各模块 docstring 中注明。本目录不做任何 import 副作用。

环境开关名保留了移植来源的项号（entry_02 清单的 P47/P85/P130/P156），
以便与上游实现及 perf_opt 文档对照；模块文件名已改为描述性名称：
flash_decode=P85、spec_draft=P130、fia_kv_refresh=P47、fused_sampler=P156。
"""
