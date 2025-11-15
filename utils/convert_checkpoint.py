#!/usr/bin/env python3
"""
命令行DCP Checkpoint转换工具

用法示例:
    # 转换为推理专用checkpoint（推荐）
    python convert_checkpoint.py \
        --dcp_path ./checkpoints/step_000012 \
        --output ./inference_checkpoints/model_step12.pth \
        --model_only

    # 转换完整checkpoint（包含优化器等）
    python convert_checkpoint.py \
        --dcp_path ./checkpoints/step_000012 \
        --output ./inference_checkpoints/full_checkpoint_step12.pth \
        --include_optimizer --include_scheduler
"""

import argparse
import sys
from pathlib import Path

from dcp_checkpoint_converter import convert_dcp_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="将DCP格式checkpoint转换为torch.save格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
转换示例:

1. 推理专用（推荐）:
   python convert_checkpoint.py \\
       --dcp_path ./checkpoints/step_000012 \\
       --output ./model_inference.pth \\
       --model_only

2. 完整checkpoint:
   python convert_checkpoint.py \\
       --dcp_path ./checkpoints/step_000012 \\
       --output ./full_checkpoint.pth \\
       --include_optimizer --include_scheduler

3. 批量转换:
   for step in 002 005 010 012; do
       python convert_checkpoint.py \\
           --dcp_path ./checkpoints/step_000$step \\
           --output ./inference/model_step$step.pth \\
           --model_only
   done
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--dcp_path", 
        type=str, 
        required=True,
        help="DCP checkpoint目录路径"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=False,
        help="输出的.pth文件路径"
    )
    
    # 选项参数
    parser.add_argument(
        "--model_only", 
        action="store_true",
        help="只保存模型和EMA权重（推理推荐，文件更小）"
    )
    
    parser.add_argument(
        "--include_optimizer", 
        action="store_true",
        help="包含优化器状态"
    )
    
    parser.add_argument(
        "--include_scheduler", 
        action="store_true",
        help="包含学习率调度器状态"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="静默模式，减少输出"
    )
    
    # 验证参数
    parser.add_argument(
        "--verify_only", 
        action="store_true",
        help="只验证DCP checkpoint是否可读，不转换"
    )
    
    args = parser.parse_args()
    
    # 验证输入路径
    dcp_path = Path(args.dcp_path)
    if not dcp_path.exists():
        print(f"❌ 错误: DCP checkpoint路径不存在: {dcp_path}")
        return 1
    
    if not list(dcp_path.glob("*.distcp")):
        print(f"❌ 错误: {dcp_path} 不是有效的DCP checkpoint（未找到.distcp文件）")
        return 1
    
    # 验证模式
    if args.verify_only:
        print(f"🔍 验证DCP checkpoint: {dcp_path}")
        try:
            import torch.distributed.checkpoint as dcp
            state_dict = {}
            dcp.load(state_dict, checkpoint_id=str(dcp_path))
            print(f"✅ DCP checkpoint有效")
            print(f"📦 包含组件: {list(state_dict.keys())}")
            return 0
        except Exception as e:
            print(f"❌ DCP checkpoint无效: {e}")
            return 1
    
    # 非验证模式需要output参数
    if not args.output:
        print("❌ 错误: 转换模式需要指定 --output 参数")
        return 1
    
    # 处理输出路径
    output_path = Path(args.output)
    if output_path.suffix != '.pth':
        print("⚠️ 警告: 建议输出文件使用.pth扩展名")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 显示转换配置
    if not args.quiet:
        print("\n" + "="*50)
        print("🔄 DCP Checkpoint转换配置")
        print("="*50)
        print(f"📥 源路径: {dcp_path}")
        print(f"📤 目标路径: {output_path}")
        print(f"📦 模式: {'推理专用' if args.model_only else '完整checkpoint'}")
        
        components = []
        if not args.model_only:
            if args.include_optimizer:
                components.append("优化器")
            if args.include_scheduler:
                components.append("调度器")
        
        if components:
            print(f"🔧 额外组件: {', '.join(components)}")
        
        print("="*50 + "\n")
    
    # 执行转换
    try:
        success = convert_dcp_checkpoint(
            dcp_path=str(dcp_path),
            output_path=str(output_path),
            model_only=args.model_only,
            include_optimizer=args.include_optimizer,
            include_scheduler=args.include_scheduler,
            verbose=not args.quiet
        )
        
        if success:
            if not args.quiet:
                print(f"\n🎉 转换成功完成!")
                print(f"📁 输出文件: {output_path}")
                print(f"💡 推理使用: checkpoint = torch.load('{output_path}')")
            return 0
        else:
            print("❌ 转换失败")
            return 1
            
    except KeyboardInterrupt:
        print("\n❌ 用户中断转换")
        return 1
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
