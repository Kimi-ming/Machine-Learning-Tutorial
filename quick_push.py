#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速提交并推送到fix分支 - 改进版
用法:
  python quick_push.py "提交信息"      # 跳过确认，直接提交
  python quick_push.py                # 交互式模式
  python quick_push.py -s "提交信息"   # 安全模式，显示改动并确认
"""

import subprocess
import sys
import os
import argparse

def run_cmd(cmd):
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    except:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='gbk', errors='ignore')
    return result

def show_status():
    """显示当前状态"""
    result = run_cmd("git status --short")
    if result.stdout:
        print("\n将要提交的改动：")
        print(result.stdout)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='快速Git提交推送工具')
    parser.add_argument('message', nargs='*', help='提交信息')
    parser.add_argument('-s', '--safe', action='store_true', help='安全模式：显示改动并确认')
    parser.add_argument('-a', '--all', action='store_true', help='添加所有文件（默认）')
    parser.add_argument('-p', '--patch', action='store_true', help='交互式选择要提交的改动')
    parser.add_argument('--skip-add', action='store_true', help='跳过git add步骤，仅提交已暂存的改动')
    args = parser.parse_args()

    # 获取提交信息
    if args.message:
        commit_msg = ' '.join(args.message)
        quick_mode = not args.safe  # 有消息且非安全模式时为快速模式
    else:
        commit_msg = input("请输入提交信息（支持中文）: ")
        if not commit_msg:
            from datetime import datetime
            commit_msg = f"更新代码 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        quick_mode = False

    # 配置git支持中文
    run_cmd("git config core.quotepath false")
    run_cmd("git config i18n.commitencoding utf-8")

    print(f"\n提交信息: {commit_msg}")

    # 显示状态并确认
    if not quick_mode:
        if show_status():
            response = input("\n确认提交这些改动？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                return 0
        else:
            print("没有需要提交的改动")
            return 0

    # 执行git命令序列
    if args.patch:
        add_cmd = "git add -p"  # 交互式添加
    elif args.skip_add:
        add_cmd = None
    else:
        add_cmd = "git add ."

    commands = [
        ("切换分支", "git checkout fix 2>/dev/null || git checkout -b fix"),
    ]
    if add_cmd:
        commands.append(("添加文件", add_cmd))
    commands.extend([
        ("提交更改", f'git commit -m "{commit_msg}"'),
        ("推送远程", "git push origin fix")
    ])

    for desc, cmd in commands:
        print(f"{desc}...", end="")
        if cmd: # Only run command if it's not None (i.e., not skipped)
            result = run_cmd(cmd)
            if result.returncode == 0 or "nothing to commit" in result.stdout:
                print(" ✓")
            elif "Everything up-to-date" in (result.stdout or "") or "Everything up-to-date" in (result.stderr or ""):
                print(" (已是最新)")
            elif "rejected" in (result.stderr or "") and "push" in cmd:
                print(" ⚠")
                print("\n检测到远程有新提交")
                print("建议操作：")
                print("1. git pull origin fix --rebase  # 拉取并变基")
                print("2. 解决可能的冲突")
                print("3. git push origin fix           # 重新推送")
                print("\n注意：避免使用 --force，会覆盖他人提交")
                return 1
            else:
                print(f" ✗\n错误: {result.stderr}")
                return 1
        else:
            print(" (跳过)")

    print("\n✅ 完成！")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(1)